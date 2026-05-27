import os
import argparse
import viser
import time
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from vggt.utils.geometry import inv
from scipy.spatial.transform import Rotation


def load_pointcloud_ply(pcd_path, color_type="rgb", max_points=5000000):
    """Load point cloud from PLY file (binary little-endian, float xyz + uchar rgb)"""
    plydata = PlyData.read(pcd_path)
    vertex = plydata['vertex']

    vertices = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(np.float32)

    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[indices]
    else:
        indices = None

    if color_type == "rgb":
        colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
        if indices is not None:
            colors = colors[indices]
        colors = colors.astype(np.float32) / 255.0
    elif color_type == "xyz":
        y_coords = vertices[:, 1].copy()
        minmax_y = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
        encoded_y = np.abs(np.sin(minmax_y * np.pi * 20.0))

        colormap = plt.get_cmap("inferno")
        colored_map = colormap(encoded_y)[:, :3]
        colors = colored_map.astype(np.float32)
    else:
        raise ValueError("Unsupported color_type. Use 'rgb' or 'xyz'.")

    return vertices, colors


def blend_colors_with_overlay(colors, overlay_color=None, colors_alpha=0.7):
    """
    colors: (..., 3) array of float32 RGB colors in [0, 1]
    """
    if overlay_color is None:
        return colors  # No overlay, return original colors

    if colors_alpha < 0 or colors_alpha > 1:
        raise ValueError("colors_alpha must be in the range [0, 1]")

    overlay = np.zeros_like(colors)
    if overlay_color == "green":
        overlay[..., 1] = 1.0
    elif overlay_color == "red":
        overlay[..., 0] = 1.0
    elif overlay_color == "blue":
        overlay[..., 2] = 1.0
    elif overlay_color == "magenta":
        overlay[..., 0] = 1.0
        overlay[..., 2] = 1.0
    else:
        raise ValueError("Unsupported overlay color. Use 'green', 'red', 'blue', or 'magenta'.")

    blended_colors = colors * colors_alpha + overlay * (1 - colors_alpha)
    return blended_colors


def load_poses_from_txt(poses_path):
    """Load c2w poses from text file. Each line is 12 floats (flattened 3x4 matrix)."""
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            pose_3x4 = np.array(values).reshape(3, 4)
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose_3x4
            poses.append(pose_4x4)
    return np.array(poses)


class SceneViewer:
    def __init__(self, server, pcds_color_type, scene_root, w2g_path=None, c2g_path=None, max_points=5000000):
        self.server = server
        self.current_scene = 0
        self.scene_objects = {}
        self.grid_height = 0.0

        self.pcds_color_type = pcds_color_type

        self.g2w_poses = None
        self.c2w_poses = None
        self.gravity_frame_visible = False
        self.camera_frame_visible = False

        self.w2g_poses = load_poses_from_txt(w2g_path)
        self.c2g_poses = load_poses_from_txt(c2g_path)
        self.g2w_poses = inv(self.w2g_poses)
        self.c2w_poses = self._compute_camera_frame_poses()

        pcd_path = os.path.join(scene_root, "pointcloud.ply")
        self.pcd_vertices, self.pcd_colors_rgb = load_pointcloud_ply(pcd_path, color_type="rgb", max_points=max_points)
        self.initial_grid_height = float(self.pcd_vertices[:, 1].max()) + 0.1

    def _compute_camera_frame_poses(self):
        """Compose g2w poses with c2g poses to get c2w poses."""
        c2w_poses = []
        for g2w, c2g in zip(self.g2w_poses, self.c2g_poses):
            c2w = g2w @ c2g
            c2w_poses.append(c2w)
        return np.array(c2w_poses)

    def _get_pcd_colors(self):
        """Derive display colors from cached point cloud data."""
        if self.pcds_color_type == "rgb":
            return self.pcd_colors_rgb
        elif self.pcds_color_type == "xyz":
            y_coords = self.pcd_vertices[:, 1].copy()
            minmax_y = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
            encoded_y = np.abs(np.sin(minmax_y * np.pi * 20.0))
            colormap = plt.get_cmap("inferno")
            return colormap(encoded_y)[:, :3].astype(np.float32)
        else:
            raise ValueError("Unsupported color_type. Use 'rgb' or 'xyz'.")

    def add_camera_frustum(self, name, c2w_pose, scale=0.3, color=(255, 0, 165)):
        """Add a camera frustum to the scene from a c2w pose."""
        position = c2w_pose[:3, 3]
        rotation_matrix = c2w_pose[:3, :3]

        r = Rotation.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        camera = self.server.scene.add_camera_frustum(
            name=name,
            fov=60.0,
            aspect=16.0 / 9.0,
            scale=scale,
            color=color,
            wxyz=quat_wxyz,
            position=position,
        )
        return camera

    def _clear_pointcloud(self):
        key = f"scene_{self.current_scene}_pcd"
        if key in self.scene_objects:
            try:
                self.scene_objects[key].remove()
            except:  # noqa
                pass
            del self.scene_objects[key]

    def _clear_frustums(self, prefix):
        for key in [k for k in self.scene_objects if k.startswith(prefix)]:
            try:
                self.scene_objects[key].remove()
            except:  # noqa
                pass
            del self.scene_objects[key]

    def clear_scene(self):
        self._clear_pointcloud()
        self._clear_frustums("gravity_frame_")
        self._clear_frustums("camera_frame_")

    def _render_pointcloud(self):
        point_cloud = self.server.scene.add_point_cloud(
            name=f"scene_{self.current_scene}_pcd",
            points=self.pcd_vertices,
            colors=self._get_pcd_colors(),
            point_size=0.0005,
        )
        self.scene_objects[f"scene_{self.current_scene}_pcd"] = point_cloud

    def _render_gravity_frustums(self):
        n = len(self.g2w_poses)
        colormap = plt.get_cmap("viridis")
        for i, pose in enumerate(self.g2w_poses):
            t = i / max(n - 1, 1)
            color = (np.array(colormap(t)[:3]) * 255).astype(np.uint8).tolist()
            camera = self.add_camera_frustum(name=f"gravity_frame_{i}", c2w_pose=pose, scale=0.01, color=color)
            self.scene_objects[f"gravity_frame_{i}"] = camera

    def _render_camera_frustums(self):
        n = len(self.c2w_poses)
        colormap = plt.get_cmap("autumn")
        for i, pose in enumerate(self.c2w_poses):
            t = i / max(n - 1, 1)
            color = (np.array(colormap(t)[:3]) * 255).astype(np.uint8).tolist()
            camera = self.add_camera_frustum(name=f"camera_frame_{i}", c2w_pose=pose, scale=0.01, color=color)
            self.scene_objects[f"camera_frame_{i}"] = camera

    def display_scene(self):
        self.clear_scene()
        self._render_pointcloud()
        if self.gravity_frame_visible and self.g2w_poses is not None:
            self._render_gravity_frustums()
        if self.camera_frame_visible and self.c2w_poses is not None:
            self._render_camera_frustums()

    def update_grid_height(self, height):
        """Update the height of the grid"""
        self.grid_height = height

        if "ground_plane" in self.scene_objects:
            try:
                self.scene_objects["ground_plane"].remove()
            except:  # noqa
                pass

        grid = self.server.scene.add_grid(
            name="ground_plane",
            width=80.0,
            height=80.0,
            width_segments=80,
            height_segments=80,
            plane='xz',
            position=(0, height, 0),
        )
        self.scene_objects["ground_plane"] = grid

    def toggle_gravity_frame_cameras(self, visible):
        self.gravity_frame_visible = visible
        self._clear_frustums("gravity_frame_")
        if visible and self.g2w_poses is not None:
            self._render_gravity_frustums()

    def toggle_camera_frame_cameras(self, visible):
        self.camera_frame_visible = visible
        self._clear_frustums("camera_frame_")
        if visible and self.c2w_poses is not None:
            self._render_camera_frustums()

    def update_color_type(self, color_type):
        self.pcds_color_type = color_type
        self._clear_pointcloud()
        self._render_pointcloud()


def main(scene_root, host="localhost", port=32942, max_points=5000000):

    w2g_path = os.path.join(scene_root, "w2g_poses.txt")
    c2g_path = os.path.join(scene_root, "c2g_poses.txt")

    server = viser.ViserServer(host=host, port=port)
    print(f"Viser server started at http://{host}:{port}")

    viewer = SceneViewer(server, "rgb", scene_root, w2g_path=w2g_path, c2g_path=c2g_path, max_points=max_points)

    with server.gui.add_folder("Point Cloud Controls"):
        color_type_dropdown = server.gui.add_dropdown(
            label="Point Cloud Color Type",
            options=["rgb", "height gradient"],
            initial_value="rgb",
        )

    with server.gui.add_folder("Camera Controls"):
        gravity_frame_checkbox = server.gui.add_checkbox(
            label="Show Gravity-Frame Frustums",
            initial_value=False,
        )
        camera_frame_checkbox = server.gui.add_checkbox(
            label="Show Camera-Frame Frustums",
            initial_value=False,
        )

    with server.gui.add_folder("Grid Controls"):
        grid_height_slider = server.gui.add_slider(
            label="Grid Height (Y)",
            min=-10.0,
            max=10.0,
            step=0.1,
            initial_value=viewer.initial_grid_height,
        )

    # Initialize the scene
    server.scene.set_up_direction('-y')
    viewer.update_grid_height(viewer.initial_grid_height)
    viewer.display_scene()

    @color_type_dropdown.on_update
    def _update_color_type(x):
        if color_type_dropdown.value == "rgb":
            color_type = "rgb"
        elif color_type_dropdown.value == "height gradient":
            color_type = "xyz"
        else:
            raise ValueError("Unsupported color type selected.")
        viewer.update_color_type(color_type)
        print(f"Switched point cloud color type to: {color_type_dropdown.value}")

    @gravity_frame_checkbox.on_update
    def _toggle_gravity_frame(x):
        viewer.toggle_gravity_frame_cameras(gravity_frame_checkbox.value)

    @camera_frame_checkbox.on_update
    def _toggle_camera_frame(x):
        viewer.toggle_camera_frame_cameras(camera_frame_checkbox.value)

    @grid_height_slider.on_update
    def _update_grid_height(x):
        viewer.update_grid_height(grid_height_slider.value)

    print("Scene visualization is ready. Use the GUI controls to interact with the scene.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


def _create_argparser():
    parser = argparse.ArgumentParser(description="Interactive 3D scene viewer for gravity-aligned point clouds.")
    parser.add_argument(
        "--scene_root", type=str, required=True,
        help="Path to the scene directory containing pointcloud.ply, w2g_poses.txt, and c2g_poses.txt."
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Host address for the viser server (default: localhost)."
    )
    parser.add_argument(
        "--port", type=int, default=27272,
        help="Port for the viser server (default: 27272)."
    )
    parser.add_argument(
        "--max_points", type=int, default=5000000,
        help="Max points to load from the point cloud (default: 5000000)."
    )
    return parser


if __name__ == "__main__":

    parser = _create_argparser()
    args = parser.parse_args()
    main(args.scene_root, host=args.host, port=args.port, max_points=args.max_points)
