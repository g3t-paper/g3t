import os
import glob
import random
import shutil
import torch
import einops
import imageio
import argparse
import numpy as np
import torch.nn.functional as F

from vggt.models.g3t import G3T
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import make_4x4, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_long.g3t_long import G3T_Long
from vggt_long.loop_utils import sim3utils
from vggt_long.loop_utils.config_utils import load_config
from vggt_long.loop_utils.sim3utils import save_confident_pointcloud_batch


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU


def write_poses(file_path, poses):

    with open(file_path, 'w') as f:
        for pose in poses:
            flat_pose = pose[:3, :].flatten()
            f.write(' '.join([str(x) for x in flat_pose]) + '\n')


def write_intrinsics(file_path, intrinsics):

    with open(file_path, 'w') as f:
        for intrinsic in intrinsics:
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            f.write(f'{fx} {fy} {cx} {cy}\n')


def extract_frames(video_path, cache_dir, nth_frame):
    os.makedirs(cache_dir, exist_ok=True)
    reader = imageio.get_reader(video_path)
    saved = 0
    for i, frame in enumerate(reader):
        if i % nth_frame == 0:
            imageio.imwrite(os.path.join(cache_dir, f"frame_{saved:06d}.png"), frame)
            saved += 1
    reader.close()
    print(f"[*] Extracted {saved} frames from input video to {cache_dir}")
    return cache_dir


def setup_model(ckpt_path):

    device = "cuda"
    if ckpt_path is None:
        model = G3T.from_pretrained("thatbrguy/g3t")

    else:
        model = G3T(
            enable_point=True, enable_depth=True,
            enable_gravity_camera_heads=True,
        )

        with open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(
            model_state_dict, strict=False
        )

    model.eval()
    model.to(device)

    return model


def run_g3t_model(images, model, points_source):

    # compute output
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            pred = model(images=images[None])

    local_pose_enc = pred["local_pose_enc"]
    global_pose_enc = pred["global_pose_enc"]

    g2c, intrinsic = pose_encoding_to_extri_intri(
        local_pose_enc, images.shape[-2:], pose_encoding_type="noT_quaR_FoV"
    )
    w2g, _ = pose_encoding_to_extri_intri(
        global_pose_enc, images.shape[-2:], pose_encoding_type="absT_quaRy_noFoV"
    )

    w2c = torch.matmul(make_4x4(g2c), make_4x4(w2g))  # w2c = g2c @ w2g
    w2c = w2c[..., :3, :].squeeze(0)  # (N, 3, 4)
    intrinsic = intrinsic.squeeze(0)  # (N, 3, 3)

    g2c = g2c.squeeze(0)  # (N, 3, 4)
    w2g = w2g.squeeze(0)  # (N, 3, 4)

    c2g = torch.zeros_like(g2c)  # (N, 3, 4) (we dont care about translation so we leave it zeros)
    c2g[:, :3, :3] = einops.rearrange(g2c[:, :3, :3], "n r c -> n c r")  # transpose rotation

    if points_source == "point_head":
        points = pred["world_points"].squeeze(0)  # (N, H, W, 3)
        conf = pred["world_points_conf"].squeeze(0)  # (N, H, W)

    elif points_source == "depth_head":
        depth = pred["depth"].squeeze(0)  # (N, H, W, 1)
        conf = pred["depth_conf"].squeeze(0)  # (N, H, W)

        points = unproject_depth_map_to_point_map(depth, w2c, intrinsic)
        points = torch.from_numpy(points).to(depth.device).to(torch.float32)  # (N, H, W, 3)

    else:
        raise ValueError(f"Invalid points_source: {points_source}")

    return points, conf, intrinsic, w2c, w2g, c2g


def run_feed_forward_inference(args, image_dir):

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup G3T Model
    model = setup_model(args.ckpt_path)

    # Get image paths and preprocess them
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"[*] No images found in {image_dir}")

    images, _ = load_and_preprocess_images_square(image_path_list, args.img_load_resolution)
    images = images.to(device)
    print(f"[*] Loaded {len(images)} images from {image_dir}")

    if len(images.shape) != 4 or images.shape[1] != 3:
        raise ValueError(f"[*] Expected images to have shape (N, 3, H, W), but got {images.shape}")

    images = F.interpolate(
        images, size=(args.inference_resolution, args.inference_resolution),
        mode="bilinear", align_corners=False
    )

    # Run G3T
    points, conf, intrinsic, w2c, w2g, c2g = run_g3t_model(
        images=images, model=model, points_source=args.points_source
    )

    # Save point cloud, intrinsics, and poses
    colors = einops.rearrange(images, "n c h w -> n h w c").cpu().numpy()  # (N, H, W, 3)
    colors = (colors * 255).astype(np.uint8)  # Scale to [0, 255] for visualization

    points = points.cpu().numpy()  # (N, H, W, 3)
    conf = conf.cpu().numpy()  # (N, H, W)
    intrinsic = intrinsic.cpu().numpy()  # (N, 3, 3)
    w2c = w2c.cpu().numpy()  # (N, 3, 4)
    w2g = w2g.cpu().numpy()  # (N, 3, 4)
    c2g = c2g.cpu().numpy()  # (N, 3, 4)

    points_path = os.path.join(args.output_dir, "pointcloud.ply")
    intrinsics_path = os.path.join(args.output_dir, "intrinsics.txt")
    w2c_path = os.path.join(args.output_dir, "w2c_poses.txt")
    w2g_path = os.path.join(args.output_dir, "w2g_poses.txt")
    c2g_path = os.path.join(args.output_dir, "c2g_poses.txt")

    save_confident_pointcloud_batch(
        points=points, colors=colors, confs=conf,
        output_path=points_path,
        conf_threshold=np.mean(conf) * args.conf_thresh_multiplier_for_viz,
    )
    write_intrinsics(intrinsics_path, intrinsic)
    write_poses(w2c_path, w2c)
    write_poses(w2g_path, w2g)
    write_poses(c2g_path, c2g)

    print(f"[*] Completed inference! Results saved to {args.output_dir}")


def run_g3t_long_inference(args, image_dir):

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setting up G3T Long config
    config = load_config(args.g3t_long_config_path)
    config["Model"]["chunk_size"] = args.chunk_size
    config["Model"]["overlap"] = args.overlap
    config["Model"]["loop_chunk_size"] = args.loop_chunk_size
    config["Model"]["loop_enable"] = args.loop_enable
    config["Model"]["model_type"] = f"g3t_{args.points_source}"
    config["Weights"]["G3T"] = args.ckpt_path
    config['Model']['Pointcloud_Save']['conf_threshold_coef'] = args.conf_thresh_multiplier_for_viz

    if config["Model"]["align_method"] == "numba":
        sim3utils.warmup_numba()

    g3t_long_obj = G3T_Long(
        image_dir, args.output_dir, config,
        conf_thresh_multiplier=args.conf_thresh_multiplier_for_alignment,
    )

    intrinsic, w2c, w2g, c2g = g3t_long_obj.run()
    g3t_long_obj.close()

    # Saving intrinsics and poses
    intrinsics_path = os.path.join(args.output_dir, "intrinsics.txt")
    w2c_path = os.path.join(args.output_dir, "w2c_poses.txt")
    w2g_path = os.path.join(args.output_dir, "w2g_poses.txt")
    c2g_path = os.path.join(args.output_dir, "c2g_poses.txt")

    write_intrinsics(intrinsics_path, intrinsic)
    write_poses(w2c_path, w2c)
    write_poses(w2g_path, w2g)
    write_poses(c2g_path, c2g)

    # Merging chunk pointclouds into one ply file
    merged_ply_path = os.path.join(args.output_dir, "pointcloud.ply")
    sim3utils.merge_ply_files(g3t_long_obj.pcd_dir, merged_ply_path)

    # Cleanup cache dir
    shutil.rmtree(g3t_long_obj.cache_dir)

    print(f"[*] Completed inference! Results saved to {args.output_dir}")


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Gravity-aligned reconstruction inference")

    # Common args
    parser.add_argument("--input", type=str, required=True,
                        help="Path to an image directory or a video file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write outputs")
    parser.add_argument("--backend", type=str, required=True,
                        choices=["feed_forward", "g3t_long"],
                        help="Inference backend to use")
    parser.add_argument("--ckpt_path", type=str,
                        help="Path to model checkpoint. Will from_pretrained to fetch weights if not provided.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (feed_forward only, default: 42)")
    parser.add_argument("--points_source", type=str, default="point_head",
                        choices=["point_head", "depth_head"],
                        help="Source for 3D points (default: point_head)")
    parser.add_argument("--conf_thresh_multiplier_for_viz", type=float, default=0.75,
                        help="Confidence threshold multiplier for visualization (default: 0.75)")

    # Video frame extraction args
    parser.add_argument("--nth_frame", type=int, default=5,
                        help="Extract every Nth frame from video (default: 5)")
    parser.add_argument("--cache_dir", type=str, default="./cache/frames",
                        help="Directory to cache extracted video frames (default: ./cache/frames)")

    # feed_forward args
    parser.add_argument("--img_load_resolution", type=int, default=1024,
                        help="Image load resolution (feed_forward only)")
    parser.add_argument("--inference_resolution", type=int, default=518,
                        help="Inference resolution (feed_forward only)")

    # g3t_long args
    parser.add_argument("--g3t_long_config_path", type=str, default="./vggt_long/configs/g3t_long.yaml",
                        help="Path to VGGT Long config file (g3t_long only)")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Chunk size for stitching (g3t_long only)")
    parser.add_argument("--overlap", type=int, default=15,
                        help="Overlap between chunks (g3t_long only)")
    parser.add_argument("--loop_chunk_size", type=int, default=7,
                        help="Loop chunk size (g3t_long only)")
    parser.add_argument("--loop_enable", action="store_true",
                        help="Enable loop closure (g3t_long only)")
    parser.add_argument("--conf_thresh_multiplier_for_alignment", type=float, default=0.1,
                        help="Confidence threshold multiplier (g3t_long only)")

    return parser
