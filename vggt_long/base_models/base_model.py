import torch
from abc import ABC, abstractmethod
from vggt.models.g3t import G3T
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.utils.geometry import make_4x4, inv

# -------------------------------------------------------
# Base class for all 3D models used in the unified pipeline.
# Every derived model must implement:
#   - load(): load model weights
#   - infer_chunk(): perform inference on a list of images
# -------------------------------------------------------

class Base3DModel(ABC):
    def __init__(self, config, device="cuda"):
        """
        Base class constructor.
        Args:
            config (dict): Configuration dictionary containing model paths/settings.
            device (str): Device to place the model on ("cuda" or "cpu").
        """
        self.config = config
        self.device = device
        # Automatically select bfloat16 for newer GPUs (SM >= 8), otherwise use float16.
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = None
        self.k = None
        self.update = True

    @abstractmethod
    def load(self):
        """Load model weights and initialize the model instance."""
        pass

    @abstractmethod
    def infer_chunk(self, image_paths: list) -> dict:
        """
        The unified inference interface used by all 3D models.
        Args:
            image_paths (list): List of image file paths.

        Returns:
            dict containing:
                - world_points: Predicted 3D points (B, N, 3)
                - world_points_conf: Confidence scores
                - images: Preprocessed input images
                - mask: Optional mask
        """
        pass


# ===================== G3T Adapter =====================
# Adapts the G3T model to the unified Base3DModel interface.
# ========================================================

class G3TAdapter(Base3DModel):
    def load(self):
        """Load G3T model and its weights into memory."""
        # Load weights specified in config
        path = self.config['Weights']['G3T']

        if path is None:
            self.model = G3T.from_pretrained("thatbrguy/g3t")
        else:
            # Initialize model
            self.model = G3T(
                enable_point=True, enable_depth=True,
                enable_gravity_camera_heads=True,
            )
            checkpoint = torch.load(path, map_location='cpu')
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.model.load_state_dict(model_state_dict, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

    def _postprocess_predictions(self, predictions):
        """
        In g3t_depth_head mode, we overwrite world_points with unprojected depth points
        We also overwrite world_points_conf with depth confidence
        """
        if self.config["Model"]["model_type"] == "g3t_point_head":

            hw = predictions["world_points"].shape[2:4]
            local_pose_enc = predictions["local_pose_enc"]
            global_pose_enc = predictions["global_pose_enc"]
            g2c, intrinsic = pose_encoding_to_extri_intri(local_pose_enc, hw, pose_encoding_type="noT_quaR_FoV")
            w2g, _ = pose_encoding_to_extri_intri(global_pose_enc, hw, pose_encoding_type="absT_quaRy_noFoV")

            predictions["intrinsic"] = intrinsic
            predictions["gravity_to_world_poses"] = inv(make_4x4(w2g))
            predictions["g2c_poses"] = make_4x4(g2c)

        elif self.config["Model"]["model_type"] == "g3t_depth_head":
            depth = predictions["depth"].squeeze(0)
            conf = predictions["depth_conf"].squeeze(0)
            hw = depth.shape[1:3]

            local_pose_enc = predictions["local_pose_enc"]
            global_pose_enc = predictions["global_pose_enc"]
            g2c, intrinsic = pose_encoding_to_extri_intri(local_pose_enc, hw, pose_encoding_type="noT_quaR_FoV")
            w2g, _ = pose_encoding_to_extri_intri(global_pose_enc, hw, pose_encoding_type="absT_quaRy_noFoV")

            extrinsic = torch.matmul(make_4x4(g2c), make_4x4(w2g))
            extrinsic = extrinsic[..., :3, :].squeeze(0)
            intrinsic = intrinsic.squeeze(0)

            points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
            points = torch.from_numpy(points).to(depth.device).to(torch.float32)

            # Update predictions with postprocessed values
            predictions["world_points"] = points.unsqueeze(0)
            predictions["world_points_conf"] = conf.unsqueeze(0)

            predictions["intrinsic"] = intrinsic.unsqueeze(0)
            predictions["gravity_to_world_poses"] = inv(make_4x4(w2g))
            predictions["g2c_poses"] = make_4x4(g2c)

        else:
            raise ValueError(f"Unknown model type: {self.config['Model']['model_type']}")

        return predictions

    def infer_chunk(self, image_paths: list) -> dict:
        """
        Run inference on a list of images using G3T.
        Handles both normal mode and "middle reference frame" mode,
        which reorders and post-processes outputs for improved consistency.
        """
        # Load images and preprocess them into a tensor: [B, 3, H, W]
        images = load_and_preprocess_images(image_paths).to(self.device)

        assert len(images.shape) == 4
        assert images.shape[1] == 3

        # Special mode: treat the middle frame as the reference image.
        if self.config['Model']['reference_frame_mid']:
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.dtype):

                    # Reorder so that the middle frame becomes the first
                    mid_idx = len(images) // 2
                    images = torch.cat(
                        [images[mid_idx:mid_idx + 1],  # middle frame
                         images[:mid_idx],             # frames before mid
                         images[mid_idx + 1:]          # frames after mid
                         ], dim=0)

                    # Run G3T
                    predictions = self.model(images)

                    # Restore original ordering back to match input order.
                    # All predicted fields must be reordered consistently.
                    predictions["world_points"] = torch.cat([
                        predictions["world_points"][:, 1:mid_idx + 1],
                        predictions["world_points"][:, :1],
                        predictions["world_points"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["world_points_conf"] = torch.cat([
                        predictions["world_points_conf"][:, 1:mid_idx + 1],
                        predictions["world_points_conf"][:, :1],
                        predictions["world_points_conf"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["images"] = torch.cat([
                        predictions["images"][:, 1:mid_idx + 1],
                        predictions["images"][:, :1],
                        predictions["images"][:, mid_idx + 1:]
                    ], dim=1)

                    if self.enable_depth:
                        predictions["depth"] = torch.cat([
                            predictions["depth"][:, 1:mid_idx + 1],
                            predictions["depth"][:, :1],
                            predictions["depth"][:, mid_idx + 1:]
                        ], dim=1)

                        predictions["depth_conf"] = torch.cat([
                            predictions["depth_conf"][:, 1:mid_idx + 1],
                            predictions["depth_conf"][:, :1],
                            predictions["depth_conf"][:, mid_idx + 1:]
                        ], dim=1)

                    if self.enable_camera:
                        predictions["pose_enc"] = torch.cat([
                            predictions["pose_enc"][:, 1:mid_idx + 1],
                            predictions["pose_enc"][:, :1],
                            predictions["pose_enc"][:, mid_idx + 1:]
                        ], dim=1)

                    if self.enable_gravity_camera_heads:
                        predictions["local_pose_enc"] = torch.cat([
                            predictions["local_pose_enc"][:, 1:mid_idx + 1],
                            predictions["local_pose_enc"][:, :1],
                            predictions["local_pose_enc"][:, mid_idx + 1:]
                        ], dim=1)

                        predictions["global_pose_enc"] = torch.cat([
                            predictions["global_pose_enc"][:, 1:mid_idx + 1],
                            predictions["global_pose_enc"][:, :1],
                            predictions["global_pose_enc"][:, mid_idx + 1:]
                        ], dim=1)

            torch.cuda.empty_cache()

        else:
            # Standard inference path
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    predictions = self.model(images)
            torch.cuda.empty_cache()

        # in g3t_depth_head mode, we overwrite world_points with unprojected depth points
        # we also overwrite world_points_conf with depth confidence
        predictions = self._postprocess_predictions(predictions)

        return {
            'world_points': predictions["world_points"],
            'world_points_conf': predictions["world_points_conf"],
            'images': predictions["images"],
            'gravity_to_world_poses': predictions["gravity_to_world_poses"],
            'g2c_poses': predictions["g2c_poses"] if "g2c_poses" in predictions else None,
            'intrinsic': predictions["intrinsic"],
            'mask': None
        }
