import gc
import os
import cv2
import glob
import torch
import einops
import shutil
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from vggt.utils.geometry import inv
from vggt_long.loop_utils import sim3utils
from vggt_long.LoopModels.LoopModel import LoopDetector
from vggt_long.base_models.base_model import G3TAdapter
from vggt_long.loop_utils.sim3loop import Sim3LoopOptimizer
from vggt_long.LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


def extract_p2_k_matrix(calib_path):
    """from calib.txt get K  (kitti)"""

    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix

    raise ValueError("P2 not found in calibration file")


class G3T_Long:
    def __init__(
        self, image_dir, save_dir, config,
        verbose=False, conf_thresh_multiplier=0.1
    ):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        # self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']
        self.model_type = self.config['Model']['model_type']
        self.observed_image_shape = None

        self.img_dir = image_dir
        self.img_list = None
        self.cache_dir = os.path.join(save_dir, "cache")
        self.verbose = verbose

        self.result_unaligned_dir = os.path.join(self.cache_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(self.cache_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(self.cache_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(self.cache_dir, 'pcd')
        self.pcd_without_conf_dir = os.path.join(self.cache_dir, 'pcd_without_conf')

        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        os.makedirs(self.pcd_without_conf_dir, exist_ok=True)

        self.all_gravity_to_world_poses = []
        self.all_camera_intrinsics = []
        self.all_g2c_poses = []

        self.delete_temp_files = self.config['Model']['delete_temp_files']

        if self.config['Weights']['model'] == 'G3T':
            self.model = G3TAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config, verbose=self.verbose)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(self.cache_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config,
                    verbose=self.verbose
                )

        if self.model_type in ("g3t_point_head", "g3t_depth_head"):
            self.rotation_dof = 1
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                "Supported types are 'g3t_point_head', 'g3t_depth_head'"
            )

        self.conf_thresh_multiplier = conf_thresh_multiplier

        if self.verbose:
            print('init done.')

    def get_loop_pairs(self):

        if self.useDBoW:  # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori  # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'],
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands  # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else:  # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # NOTE: self.observed_image_shape assumes that all images have the same shape,
        # which is true for the eval datasets that we use with VGGT Long

        # Record the observed image shape (H, W) once from the first chunk
        if self.observed_image_shape is None and "images" in predictions:
            # predictions["images"] has shape (N, 3, H, W) after squeeze
            self.observed_image_shape = np.array(predictions["images"].shape[2:])

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            gravity_to_world_poses = predictions['gravity_to_world_poses']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_gravity_to_world_poses.append((chunk_range, gravity_to_world_poses))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))
            self.all_g2c_poses.append((chunk_range, predictions['g2c_poses']))
        # predictions['depth'] = np.squeeze(predictions['depth'])

        np.save(save_path, predictions)

        return predictions if is_loop or range_2 is not None else None

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) must be less "
                f"than chunk size ({self.chunk_size})"
            )
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        tqdm_msg = "[*] Feed-forward overlapping chunks"
        for chunk_idx in tqdm(range(len(self.chunk_indices)), desc=tqdm_msg):
            if self.verbose:
                print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        if self.loop_enable:

            print('[*] Detecting loop closures')
            loop_results = sim3utils.process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window = int(self.config['Model']['loop_chunk_size'] / 2)
            )
            loop_results = remove_duplicates(loop_results)
            if self.verbose:
                print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))

            tqdm_msg = "[*] Feed-forward loop closure chunks"
            for item in tqdm(loop_results, desc=tqdm_msg):
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                self.loop_predict_list.append((item, single_chunk_predictions))
                if self.verbose:
                    print(item)

        del self.model  # Save GPU Memory
        torch.cuda.empty_cache()

        if self.verbose:
            print("Aligning all the chunks...")

        print("[*] Performing chunk alignment")
        for chunk_idx in range(len(self.chunk_indices)-1):

            if self.verbose:
                print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True
            ).item()
            chunk_data2 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True
            ).item()

            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]

            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-self.overlap:]
                mask2 = chunk_data2["mask"][:self.overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            conf_threshold = min(np.median(conf1), np.median(conf2)) * self.conf_thresh_multiplier

            s, R, t = sim3utils.weighted_align_point_maps(
                point_map1,
                conf1,
                point_map2,
                conf2,
                mask,
                conf_threshold=conf_threshold,
                config=self.config,
                rotation_dof=self.rotation_dof
            )

            if self.verbose:
                print("Estimated Scale:", s)
                print("Estimated Rotation:\n", R)
                print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))

        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                if self.verbose:
                    print('chunk_a align')
                point_map_loop = item[1]['world_points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['world_points_conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                if self.verbose:
                    print(self.chunk_indices[chunk_idx_a])
                    print(chunk_a_range)
                    print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True
                ).item()

                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]

                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * self.conf_thresh_multiplier
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][:chunk_a_range[1] - chunk_a_range[0]]
                    mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                    mask = mask_loop.squeeze() & mask_a.squeeze()

                s_a, R_a, t_a = sim3utils.weighted_align_point_maps(
                    point_map_a,
                    conf_a,
                    point_map_loop,
                    conf_loop,
                    mask,
                    conf_threshold=conf_threshold,
                    config=self.config,
                    rotation_dof=self.rotation_dof
                )

                if self.verbose:
                    print("Estimated Scale:", s_a)
                    print("Estimated Rotation:\n", R_a)
                    print("Estimated Translation:", t_a)

                    print('chunk_a align')
                point_map_loop = item[1]['world_points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['world_points_conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                if self.verbose:
                    print(self.chunk_indices[chunk_idx_b])
                    print(chunk_b_range)
                    print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                    allow_pickle=True
                ).item()

                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]

                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * self.conf_thresh_multiplier
                mask = None
                if item[1]['mask'] is not None:
                    mask_loop = item[1]['mask'][-chunk_b_range[1] + chunk_b_range[0]:]
                    mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                    mask = mask_loop.squeeze() & mask_b.squeeze()

                s_b, R_b, t_b = sim3utils.weighted_align_point_maps(
                    point_map_b,
                    conf_b,
                    point_map_loop,
                    conf_loop,
                    mask,
                    conf_threshold=conf_threshold,
                    config=self.config,
                    rotation_dof=self.rotation_dof
                )

                if self.verbose:
                    print("Estimated Scale:", s_b)
                    print("Estimated Rotation:\n", R_b)
                    print("Estimated Translation:", t_b)

                    print('a -> b SIM 3')
                s_ab, R_ab, t_ab = sim3utils.compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                if self.verbose:
                    print("Estimated Scale:", s_ab)
                    print("Estimated Rotation:\n", R_ab)
                    print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        if self.loop_enable:
            print("[*] Performing loop closure optimization")

            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(
                self.sim3_list, self.loop_sim3_list,
                rotation_dof=self.rotation_dof
            )
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]

            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visual in png format
            plt.figure(figsize=(8, 6))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.cache_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        if self.verbose:
            print('Apply alignment')

        print("[*] Post-processing results")
        self.sim3_list_without_acc = deepcopy(self.sim3_list)
        self.sim3_list = sim3utils.accumulate_sim3_transforms(self.sim3_list)

        for chunk_idx in range(len(self.chunk_indices) - 1):
            if self.verbose:
                print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx + 1}.npy"),
                allow_pickle=True
            ).item()

            chunk_data['world_points'] = sim3utils.apply_sim3_direct(chunk_data['world_points'], s, R, t)

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:

                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                    allow_pickle=True
                ).item()

                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

                points_first = chunk_data_first['world_points'].reshape(-1, 3)
                colors_first = (chunk_data_first['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
                confs_first = chunk_data_first['world_points_conf'].reshape(-1)
                ply_path_first = os.path.join(self.pcd_dir, '0_pcd.ply')
                ply_path_first_no_conf = os.path.join(self.pcd_without_conf_dir, '0_pcd_no_conf.ply')
                sim3utils.save_pointcloud(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    output_path=ply_path_first_no_conf,
                )
                sim3utils.save_confident_pointcloud_batch(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    confs=confs_first,  # shape: (H, W)
                    output_path=ply_path_first,
                    conf_threshold=np.mean(confs_first) * self.config['Model']['Pointcloud_Save'][
                        'conf_threshold_coef'],
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
                )

            aligned_chunk_data = chunk_data

            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx + 1}_pcd.ply')
            ply_path_no_conf = os.path.join(self.pcd_without_conf_dir, f'{chunk_idx + 1}_pcd_no_conf.ply')
            sim3utils.save_pointcloud(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                output_path=ply_path_no_conf,
            )
            sim3utils.save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        intrinsic, w2c, w2g, c2g = self.collect_intrinsics_and_poses()

        if self.verbose:
            print('Done.')

        return intrinsic, w2c, w2g, c2g

    def run(self):
        if self.verbose:
            print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        if self.verbose:
            print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close()  # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector  # Save GPU Memory
        torch.cuda.empty_cache()
        if self.verbose:
            print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        intrinsic, w2c, w2g, c2g = self.process_long_sequence()
        return intrinsic, w2c, w2g, c2g

    def collect_intrinsics_and_poses(self):
        '''
        Collects intrinsics, w2c, w2g, c2g
        '''
        all_g2c_poses = [None] * len(self.img_list)
        all_g2w_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_g2w = self.all_gravity_to_world_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        _, first_chunk_g2c = self.all_g2c_poses[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            g2w = first_chunk_g2w[i]
            all_g2w_poses[idx] = g2w
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]
            if first_chunk_g2c is not None:
                all_g2c_poses[idx] = first_chunk_g2c[i]

        for chunk_idx in range(1, len(self.all_gravity_to_world_poses)):
            chunk_range, chunk_g2w = self.all_gravity_to_world_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            _, chunk_g2c = self.all_g2c_poses[chunk_idx]
            # When call self.save_gravity_to_world_poses(), all the poses are aligned to the first chunk.
            s, R, t = self.sim3_list[chunk_idx - 1]

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                g2w = chunk_g2w[i]  #

                transformed_g2w = S @ g2w  # Be aware of the left multiplication!
                transformed_g2w[:3, :3] /= s  # Normalize rotation

                all_g2w_poses[idx] = transformed_g2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]
                if chunk_g2c is not None:
                    all_g2c_poses[idx] = chunk_g2c[i]

        all_g2c_poses = np.stack(all_g2c_poses, axis=0)
        all_g2w_poses = np.stack(all_g2w_poses, axis=0)
        all_intrinsics = np.stack(all_intrinsics, axis=0)

        all_w2c_poses = np.matmul(all_g2c_poses, inv(all_g2w_poses))
        all_c2g_poses = np.zeros_like(all_g2c_poses)
        all_c2g_poses[:, :3, :3] = einops.rearrange(all_g2c_poses[:, :3, :3], "n r c -> n c r")

        return all_intrinsics, all_w2c_poses, all_g2w_poses, all_c2g_poses

    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.

            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results

            ~50 GiB for 4500-frame KITTI 00,
            ~35 GiB for 2700-frame KITTI 05,
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return

        total_space = 0

        if self.verbose:
            print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        if self.verbose:
            print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        if self.verbose:
            print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        if self.verbose:
            print('Deleting temp files done.')

            print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':
    pass
