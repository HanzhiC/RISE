from easydict import EasyDict as edict
import numpy as np
import open3d as o3d
import viser
import viser.transforms as tf
import time
import threading
from typing import List, Optional, Dict, Any
from easydict import EasyDict as edict

T_z_m90 = np.array(
    [
        [0, -1, 0, 0],  # cos(90), -sin(90), 0, 0
        [1, 0, 0, 0],  # sin(90),  cos(90), 0, 0
        [0, 0, 1, 0],  # 0,        0,       1, 0
        [0, 0, 0, 1],  # 0,        0,       0, 1
    ]
).T





class O3dSceneFlowViewer:
    """
    Enhanced Open3D visualizer for scene flow visualization with SLAM-based camera control.
    """

    def __init__(
        self,
        vis_scenes,
        vis_trajs,
        slam_poses,
        pcd_slam=None,
        front_distance=1.5,
        viewer_name="Scene Flow Viewer",
        port=None,
        point_size=0.005,
    ):
        """
        Initialize the SceneFlowViewer.

        Args:
            vis_scenes: List of scene geometries for each frame
            vis_trajs: List of trajectory geometries for each frame
            slam_poses: List of SLAM pose matrices
            pcd_slam: Point cloud for SLAM visualization
            vis_wrist_traj: Wrist trajectory visualization
            vis_palm_traj: Palm trajectory visualization
            vis_approch_vec: Approach vector visualization
        """
        self.vis_scenes = vis_scenes
        self.vis_trajs = vis_trajs
        self.preprocess_vis_trajs(vis_trajs)
        self.vis_cameras = []
        # for ii in range(len(slam_poses)):
        #     T_world_cam = slam_poses[ii]
        #     vis_camera_frustum = visualize_camera_frustum(
        #         T_world_cam @ T_z_m90,
        #         scale=0.2,
        #         color=np.array([0.8, 0.2, 0.2]),
        #         return_mesh=True,
        #     )
        #     self.vis_cameras.append(vis_camera_frustum)
        self.pcd_slam = pcd_slam
        self.slam_poses = slam_poses
        self.point_size = point_size
        # Initialize visualization parameters
        self.vis_param = edict()
        self.vis_param.n_steps_left = 0
        self.vis_param.index = 0
        self.vis_param.n_imgs = len(vis_scenes)
        self.vis_param.stop = True
        self.vis_param.playing = False
        self.vis_param.speed = 1
        self.vis_param.show_slam = True if pcd_slam is not None else False
        self.vis_param.show_scene_flow = True
        self.vis_param.follow_slam = True
        self.vis_param.slam_points_visible = self.vis_param.show_slam
        self.vis_param.auto_camera_update = (
            True  # New flag to control automatic camera updates
        )
        self.viewer_name = viewer_name
        # Initialize visualizer
        self.window = None
        self.camera_state = None
        self.front_distance = front_distance
        print(
            f"Set Front Distance to -10 if you are inputting whole scene pointclouds as pcd_slam when viewing Aria dataset"
        )

    def preprocess_vis_trajs(self, vis_trajs):
        if isinstance(vis_trajs[0], List):
            new_vis_trajs = []
            for vis_traj in vis_trajs:
                vis_traj_mesh = o3d.geometry.TriangleMesh()
                for vis_traj_point in vis_traj:
                    vis_traj_mesh += vis_traj_point
                new_vis_trajs.append(vis_traj_mesh)
            self.vis_trajs = new_vis_trajs
        else:
            self.vis_trajs = vis_trajs

    def set_camera_from_current_pose(self, frame_idx):
        """Set camera viewpoint based on current pose"""
        if frame_idx >= len(self.vis_scenes):
            return
        self.set_camera_from_slam_pose(self.slam_poses[frame_idx], frame_idx)

    def set_camera_from_slam_pose(self, slam_pose, frame_idx):
        """Set camera viewpoint based on SLAM pose with 90-degree Z rotation"""
        if slam_pose is None or frame_idx >= len(self.slam_poses):
            return

        # Extract camera position and orientation from SLAM pose
        T_world_cam = self.slam_poses[frame_idx]

        # Camera position is the translation part of the transformation matrix
        camera_pos = T_world_cam[:3, 3]

        # Camera orientation: extract rotation matrix and convert to front/up vectors
        R_cam_world = T_world_cam[:3, :3]

        # Create 90-degree rotation around Z axis
        # This rotates the camera coordinate system
        R_z_90 = np.array(
            [
                [0, -1, 0],  # cos(90), -sin(90), 0
                [1, 0, 0],  # sin(90),  cos(90), 0
                [0, 0, 1],  # 0,        0,       1
            ]
        )

        # Apply the Z rotation to the camera orientation
        R_cam_world_rotated = R_cam_world @ R_z_90

        # Camera looks in the negative Z direction in camera coordinates
        # So front vector is the negative of the Z axis in world coordinates
        front_vector = -R_cam_world_rotated[:, 2]  # Negative Z axis
        up_vector = R_cam_world_rotated[:, 1]  # Y axis

        # # Set a point slightly in front of the camera to look at
        # if self.pcd_slam is not None:
        #     camera_pos = camera_pos - front_vector * 10.0
        # else:
        camera_pos = camera_pos + front_vector * self.front_distance
        look_at_point = camera_pos - front_vector * 2.0

        # Get view control and set camera parameters
        ctr = self.window.get_view_control()
        ctr.set_front(front_vector)
        ctr.set_lookat(look_at_point)
        ctr.set_up(up_vector)
        ctr.set_zoom(0.3)  # Adjust zoom as needed

    def toggle_play(self, vis):
        self.vis_param.playing = not self.vis_param.playing
        if self.vis_param.playing:
            self.vis_param.n_steps_left = 10000
        print(f"Playing: {self.vis_param.playing}")
        return False

    def step_forward(self, vis):
        self.vis_param.n_steps_left = 1
        # Update camera if following SLAM
        if self.vis_param.follow_slam and self.vis_param.index < len(self.slam_poses):
            self.set_camera_from_slam_pose(
                self.slam_poses[self.vis_param.index], self.vis_param.index
            )
        return False

    def step_backward(self, vis):
        self.vis_param.index = max(0, self.vis_param.index - 1)
        self.vis_param.n_steps_left = 0
        # Clear current frame and redraw
        if self.vis_param.index > 0:
            self.window.remove_geometry(self.vis_scenes[self.vis_param.index], False)
            self.window.remove_geometry(self.vis_trajs[self.vis_param.index], False)
            # self.window.remove_geometry(self.vis_cameras[self.vis_param.index], False)
            if self.vis_param.index > 1:
                self.window.add_geometry(
                    self.vis_trajs[self.vis_param.index - 1], False
                )
                self.window.add_geometry(
                    self.vis_scenes[self.vis_param.index - 1], False
                )
                # self.window.add_geometry(
                #     self.vis_cameras[self.vis_param.index - 1], False
                # )

        # Update camera if following SLAM
        if self.vis_param.follow_slam and self.vis_param.index < len(self.slam_poses):
            self.set_camera_from_slam_pose(
                self.slam_poses[self.vis_param.index], self.vis_param.index
            )

        print(f"Frame: {self.vis_param.index}")
        return False

    def toggle_slam_points(self, vis):
        self.vis_param.show_slam = not self.vis_param.show_slam
        # Remove or add SLAM points
        if not self.vis_param.show_slam and self.vis_param.slam_points_visible:
            print("Removing SLAM points")
            if self.pcd_slam is not None:
                self.window.remove_geometry(self.pcd_slam)
            self.vis_param.slam_points_visible = False
        elif self.vis_param.show_slam and not self.vis_param.slam_points_visible:
            if self.pcd_slam is not None:
                self.window.add_geometry(self.pcd_slam)
            self.vis_param.slam_points_visible = True
        print(f"SLAM points: {'ON' if self.vis_param.show_slam else 'OFF'}")
        return False

    def toggle_auto_camera_update(self, vis):
        """Toggle automatic camera updates"""
        self.vis_param.auto_camera_update = not self.vis_param.auto_camera_update
        print(
            f"Auto camera update: {'ON' if self.vis_param.auto_camera_update else 'OFF'}"
        )
        return False

    def reset_view(self, vis):
        # Reset camera to default view
        ctr = self.window.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.7)
        return False

    def print_help(self, vis):
        print("\n=== Interactive Viewer Controls ===")
        print("SPACE: Toggle play/pause")
        print("RIGHT ARROW: Step forward")
        print("LEFT ARROW: Step backward")
        print("L: Toggle SLAM points")
        print("C: Toggle SLAM camera following")
        print("A: Toggle auto camera update")
        print("R: Reset camera view")
        print("H: Show this help")
        print("ESC: Exit")
        print("================================")
        return False

    def drawNextFrame(self, vis):
        if self.vis_param.n_steps_left <= 0 or self.vis_param.index >= len(
            self.vis_scenes
        ):
            return False

        # Add current frame
        self.window.add_geometry(self.vis_trajs[self.vis_param.index])
        self.window.add_geometry(self.vis_scenes[self.vis_param.index])
        # self.window.add_geometry(self.vis_cameras[self.vis_param.index])
        if self.vis_param.index > 0:
            self.window.remove_geometry(self.vis_trajs[self.vis_param.index - 1], False)
            self.window.remove_geometry(
                self.vis_scenes[self.vis_param.index - 1], False
            )
            # self.window.remove_geometry(
            #     self.vis_cameras[self.vis_param.index - 1], False
            # )
        self.vis_param.n_steps_left -= 1
        self.vis_param.index += 1

        # Update camera if following SLAM and auto camera update is enabled
        if self.vis_param.follow_slam and self.vis_param.auto_camera_update:
            vis_idx = min(self.vis_param.index, len(self.slam_poses) - 1)
            self.set_camera_from_slam_pose(self.slam_poses[vis_idx], vis_idx)

        # Print frame information instead of updating window title
        print(f"Frame: {self.vis_param.index}/{self.vis_param.n_imgs}")

        return True

    def run(self):
        """Run the visualizer"""
        # Create the enhanced visualizer
        self.window = o3d.visualization.VisualizerWithKeyCallback()
        self.window.create_window(
            window_name=f"{self.viewer_name}", width=1280, height=720, visible=True
        )

        # Configure render options
        render_option = self.window.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.point_size = 5.0
        render_option.background_color = np.array([0.5, 0.5, 0.5])  # Grey background

        # Add static geometries
        if self.pcd_slam is not None:
            self.window.add_geometry(self.pcd_slam)

        # Register keyboard callbacks
        self.window.register_key_callback(
            key=ord(" "), callback_func=self.toggle_play
        )  # Space
        self.window.register_key_callback(
            key=ord("."), callback_func=self.toggle_slam_points
        )  # L
        self.window.register_key_callback(
            key=ord("a"), callback_func=self.toggle_auto_camera_update
        )  # A
        self.window.register_key_callback(
            key=ord("r"), callback_func=self.reset_view
        )  # R
        self.window.register_key_callback(
            key=ord("h"), callback_func=self.print_help
        )  # H

        # Arrow key callbacks
        self.window.register_key_callback(
            key=262, callback_func=self.step_forward
        )  # Right arrow
        self.window.register_key_callback(
            key=263, callback_func=self.step_backward
        )  # Left arrow

        # Register animation callback
        self.window.register_animation_callback(callback_func=self.drawNextFrame)

        # Set initial camera position based on first SLAM pose
        if self.vis_param.follow_slam and len(self.slam_poses) > 0:
            self.set_camera_from_slam_pose(self.slam_poses[0], 0)

        # Print initial help
        self.print_help(None)

        # Run the visualizer
        self.window.run()
        self.window.destroy_window()


class ViserSceneFlowViewer:
    """
    Web-based scene flow visualizer using Viser with SLAM-based camera control.
    Equivalent functionality to the Open3D SceneFlowViewer.
    """

    def __init__(
        self,
        vis_scenes: List[Any],  # List of scene geometries for each frame
        vis_trajs: List[Any],  # List of trajectory geometries for each frame
        slam_poses: List[np.ndarray],  # List of SLAM pose matrices
        pcd_slam: Optional[Any] = None,  # Point cloud for SLAM visualization
        front_distance: float = 0.0,
        viewer_name: str = "Scene Flow Viewer",
        port: int = 8080,
        point_size: float = 0.005,
    ):
        """
        Initialize the ViserSceneFlowViewer.

        Args:
            vis_scenes: List of scene geometries for each frame
            vis_trajs: List of trajectory geometries for each frame
            slam_poses: List of SLAM pose matrices
            pcd_slam: Point cloud for SLAM visualization
            front_distance: Distance to position camera in front of SLAM pose
            viewer_name: Name for the viewer window
            port: Port for the web server
        """
        self.vis_scenes = vis_scenes
        self.vis_trajs = vis_trajs
        self.slam_poses = slam_poses
        self.pcd_slam = pcd_slam
        self.front_distance = front_distance
        self.viewer_name = viewer_name
        self.point_size = point_size
        # Initialize visualization parameters
        self.vis_param = edict()
        self.vis_param.index = 0
        self.vis_param.n_imgs = len(vis_scenes)
        self.vis_param.playing = False
        self.vis_param.speed = 1
        self.vis_param.show_slam = True if pcd_slam is not None else False
        self.vis_param.show_scene_flow = True
        self.vis_param.follow_slam = True
        self.vis_param.auto_camera_update = True

        # Initialize viser server
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")

        # Store handles for dynamic objects
        self.scene_handles = []
        self.traj_handles = []
        self.camera_handles = []
        self.slam_handle = None

        # Animation thread
        self.animation_thread = None
        self.stop_animation = False
        print(
            f"Set Front Distance to -10 if you are inputting whole scene pointclouds as pcd_slam when viewing Aria dataset"
        )
        print(f"Web viewer available at: http://localhost:{port}")

    def _convert_o3d_to_viser_data(self, o3d_geometry):
        """Convert Open3D geometry to viser-compatible data"""
        if hasattr(o3d_geometry, "points"):
            # Point cloud
            points = np.asarray(o3d_geometry.points)
            colors = None
            if o3d_geometry.has_colors():
                colors = np.asarray(o3d_geometry.colors)
            else:
                colors = np.ones_like(points) * 0.5
            return points, colors
        elif hasattr(o3d_geometry, "vertices"):
            # Mesh
            vertices = np.asarray(o3d_geometry.vertices)
            colors = None
            if o3d_geometry.has_vertex_colors():
                colors = np.asarray(o3d_geometry.vertex_colors)
            else:
                colors = np.ones_like(vertices) * 0.5
            return vertices, colors
        return None, None

    def _setup_gui_controls(self):
        """Setup GUI controls for the viewer"""
        with self.server.add_gui_folder("Scene Flow Controls"):
            # Frame navigation
            self.frame_slider = self.server.add_gui_slider(
                "Frame",
                min=0,
                max=self.vis_param.n_imgs - 1,
                step=1,
                initial_value=0,
            )

            # Playback controls
            self.play_button = self.server.add_gui_button("Play/Pause")
            self.step_forward_button = self.server.add_gui_button("Step Forward")
            self.step_backward_button = self.server.add_gui_button("Step Backward")

            # Speed control
            self.speed_slider = self.server.add_gui_slider(
                "Speed",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=0.5,
            )

            # Visibility toggles
            self.show_slam_checkbox = self.server.add_gui_checkbox(
                "Show SLAM Points",
                initial_value=self.vis_param.show_slam,
            )

            self.show_scene_flow_checkbox = self.server.add_gui_checkbox(
                "Show Scene Flow",
                initial_value=self.vis_param.show_scene_flow,
            )

            self.follow_slam_checkbox = self.server.add_gui_checkbox(
                "Follow SLAM Camera",
                initial_value=self.vis_param.follow_slam,
            )

            self.auto_camera_update_checkbox = self.server.add_gui_checkbox(
                "Auto Camera Update",
                initial_value=self.vis_param.auto_camera_update,
            )

            # Camera controls
            self.reset_view_button = self.server.add_gui_button("Reset View")

            # Point size control
            self.point_size_slider = self.server.add_gui_slider(
                "Point Size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=self.point_size,
            )

    def _setup_gui_callbacks(self):
        """Setup callbacks for GUI controls"""

        @self.frame_slider.on_update
        def _(_):
            self.vis_param.index = self.frame_slider.value
            self._update_frame_display()
            if self.vis_param.follow_slam and self.vis_param.auto_camera_update:
                self._set_camera_from_slam_pose(
                    self.slam_poses[self.vis_param.index], self.vis_param.index
                )

        @self.play_button.on_click
        def _(_):
            self.vis_param.playing = not self.vis_param.playing
            if self.vis_param.playing:
                self._start_animation()
            else:
                self._stop_animation()

        @self.step_forward_button.on_click
        def _(_):
            if self.vis_param.index < self.vis_param.n_imgs - 1:
                self.vis_param.index += 1
                self.frame_slider.value = self.vis_param.index
                self._update_frame_display()
                if self.vis_param.follow_slam and self.vis_param.auto_camera_update:
                    self._set_camera_from_slam_pose(
                        self.slam_poses[self.vis_param.index], self.vis_param.index
                    )

        @self.step_backward_button.on_click
        def _(_):
            if self.vis_param.index > 0:
                self.vis_param.index -= 1
                self.frame_slider.value = self.vis_param.index
                self._update_frame_display()
                if self.vis_param.follow_slam and self.vis_param.auto_camera_update:
                    self._set_camera_from_slam_pose(
                        self.slam_poses[self.vis_param.index], self.vis_param.index
                    )

        @self.speed_slider.on_update
        def _(_):
            self.vis_param.speed = self.speed_slider.value

        @self.show_slam_checkbox.on_update
        def _(_):
            self.vis_param.show_slam = self.show_slam_checkbox.value
            self._update_slam_visibility()

        @self.show_scene_flow_checkbox.on_update
        def _(_):
            self.vis_param.show_scene_flow = self.show_scene_flow_checkbox.value
            self._update_scene_flow_visibility()

        @self.follow_slam_checkbox.on_update
        def _(_):
            self.vis_param.follow_slam = self.follow_slam_checkbox.value

        @self.auto_camera_update_checkbox.on_update
        def _(_):
            self.vis_param.auto_camera_update = self.auto_camera_update_checkbox.value

        @self.reset_view_button.on_click
        def _(_):
            self._reset_camera_view()

        @self.point_size_slider.on_update
        def _(_):
            self._update_point_sizes()

    def _set_camera_from_slam_pose(self, slam_pose: np.ndarray, frame_idx: int):
        """Set camera viewpoint based on SLAM pose with 90-degree Z rotation"""
        if slam_pose is None or frame_idx >= len(self.slam_poses):
            return

        # Extract camera position and orientation from SLAM pose
        T_world_cam = self.slam_poses[frame_idx]

        # Camera position is the translation part of the transformation matrix
        camera_pos = T_world_cam[:3, 3]

        # Camera orientation: extract rotation matrix and convert to front/up vectors
        R_cam_world = T_world_cam[:3, :3]

        # Create 90-degree rotation around Z axis
        R_z_90 = np.array(
            [
                [0, -1, 0],  # cos(90), -sin(90), 0
                [1, 0, 0],  # sin(90),  cos(90), 0
                [0, 0, 1],  # 0,        0,       1
            ]
        )

        # Apply the Z rotation to the camera orientation
        R_cam_world_rotated = R_cam_world @ R_z_90.T

        # Camera looks in the negative Z direction in camera coordinates
        front_vector = -R_cam_world_rotated[:, 2]  # Negative Z axis
        up_vector = R_cam_world_rotated[:, 1]  # Y axis

        # Position camera slightly in front
        camera_pos = camera_pos + front_vector * self.front_distance
        look_at_point = camera_pos - front_vector * 2.0

        # Convert to viser format
        # Create rotation matrix from front and up vectors
        right_vector = np.cross(front_vector, up_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)
        up_vector = np.cross(right_vector, front_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)

        # Create rotation matrix (world to camera)
        R_world_cam = np.column_stack([right_vector, up_vector, -front_vector])

        # Convert to quaternion
        wxyz = tf.SO3.from_matrix(R_world_cam).wxyz

        # Set camera for all connected clients
        for client in self.server.get_clients().values():
            client.camera.wxyz = wxyz
            client.camera.position = camera_pos

    def _reset_camera_view(self):
        """Reset camera to default view"""
        for client in self.server.get_clients().values():
            client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            client.camera.position = np.array([0.0, 0.0, 5.0])
            client.camera.fov = 45.0

    def _update_frame_display(self):
        """Update the display to show current frame"""
        with self.server.atomic():
            # Hide all frames
            for i, handle in enumerate(self.scene_handles):
                handle.visible = i == self.vis_param.index

            for i, handle in enumerate(self.traj_handles):
                handle.visible = i == self.vis_param.index

    def _update_slam_visibility(self):
        """Update SLAM points visibility"""
        if self.slam_handle is not None:
            self.slam_handle.visible = self.vis_param.show_slam

    def _update_scene_flow_visibility(self):
        """Update scene flow visibility"""
        for i, handle in enumerate(self.scene_handles):
            handle.visible = self.vis_param.show_scene_flow and (
                i == self.vis_param.index
            )

    def _update_point_sizes(self):
        """Update point sizes for all point clouds"""
        point_size = self.point_size_slider.value

        # Update SLAM points
        if self.slam_handle is not None:
            self.slam_handle.point_size = point_size

        # Update scene points
        for handle in self.scene_handles:
            handle.point_size = point_size

        # Update trajectory points
        for handle in self.traj_handles:
            handle.point_size = point_size

    def _start_animation(self):
        """Start the animation thread"""
        if self.animation_thread is None or not self.animation_thread.is_alive():
            self.stop_animation = False
            self.animation_thread = threading.Thread(target=self._animation_loop)
            self.animation_thread.daemon = True
            self.animation_thread.start()

    def _stop_animation(self):
        """Stop the animation thread"""
        self.stop_animation = True

    def _animation_loop(self):
        """Animation loop running in separate thread"""
        while not self.stop_animation and self.vis_param.playing:
            if self.vis_param.index < self.vis_param.n_imgs - 1:
                self.vis_param.index += 1
                self.frame_slider.value = self.vis_param.index
                self._update_frame_display()

                if self.vis_param.follow_slam and self.vis_param.auto_camera_update:
                    self._set_camera_from_slam_pose(
                        self.slam_poses[self.vis_param.index], self.vis_param.index
                    )
            else:
                self.vis_param.playing = False
                break

            time.sleep(1.0 / (30.0 * self.vis_param.speed))  # 30 FPS base

    def _add_scene_geometry(self, frame_idx: int, geometry):
        """Add scene geometry for a specific frame"""
        points, colors = self._convert_o3d_to_viser_data(geometry)
        if points is not None:
            handle = self.server.add_point_cloud(
                name=f"/frames/{frame_idx}/scene",
                points=points,
                colors=colors,
                point_size=0.0025,
                visible=(frame_idx == self.vis_param.index),
                point_shape="circle",
            )
            self.scene_handles.append(handle)

    def _add_trajectory_geometry(self, frame_idx: int, geometry_list):
        """Add trajectory geometry for a specific frame"""
        points, colors = [], []
        for geometry in geometry_list:
            points.append(np.asarray(geometry.vertices).mean(axis=0))
            colors.append(np.asarray(geometry.vertex_colors).mean(axis=0))
        
        points = np.stack(points, axis=0)
        colors = np.stack(colors, axis=0)
        handle = self.server.add_point_cloud(
            name=f"/frames/{frame_idx}/trajectory",
            points=points,
            colors=colors,
            point_size=0.01,
            visible=(frame_idx == self.vis_param.index),
            point_shape="circle",
        )
        self.traj_handles.append(handle)

    def _add_camera_frustum(self, frame_idx: int, pose):
        T_world_cam = pose
        T_world_cam_rotated = T_world_cam @ T_z_m90
        position = T_world_cam_rotated[:3, 3]
        rotation_matrix = T_world_cam_rotated[:3, :3]
        wxyz = tf.SO3.from_matrix(rotation_matrix).wxyz

        handle = self.server.add_camera_frustum(
            name=f"/frames/{frame_idx}/camera",
            fov=45.0,
            aspect=1.0,
            wxyz=wxyz,
            position=position,
            scale=0.1,
            visible=(frame_idx == self.vis_param.index),
            color=(204, 51, 51),  # Red color like in original
            image=None,
        )

    def _add_slam_pointcloud(self):
        """Add SLAM point cloud"""
        if self.pcd_slam is not None:
            points, colors = self._convert_o3d_to_viser_data(self.pcd_slam)

            if points is not None:
                self.slam_handle = self.server.add_point_cloud(
                    name="/slam/points",
                    points=points,
                    colors=colors,
                    point_size=self.point_size_slider.value,
                    visible=self.vis_param.show_slam,
                )

    # def _add_camera_frustums(self):
    #     """Add camera frustums for SLAM poses"""
    #     for i, slam_pose in enumerate(self.slam_poses):
    #         T_world_cam = slam_pose

    #         # Apply Z rotation like in the original code
    #         T_world_cam_rotated = T_world_cam @ T_z_m90

    #         # Extract position and rotation
    #         position = T_world_cam_rotated[:3, 3]
    #         rotation_matrix = T_world_cam_rotated[:3, :3]
    #         wxyz = tf.SO3.from_matrix(rotation_matrix).wxyz

    #         handle = self.server.add_camera_frustum(
    #             name=f"/cameras/{i}",
    #             fov=45.0,  # Default FOV
    #             aspect=0.5,
    #             wxyz=wxyz,
    #             position=position,
    #             scale=0.1,
    #             color=(204, 51, 51),  # Red color like in original
    #             visible=(frame_idx == self.vis_param.index),
    #         )
    #         self.camera_handles.append(handle)

    def _setup_scene(self):
        """Setup the 3D scene with all geometries"""
        # Create frames folder
        self.server.add_frame("/frames", show_axes=False)

        # Add SLAM point cloud
        self._add_slam_pointcloud()

        # Add scene geometries for each frame
        for i, scene_geom in enumerate(self.vis_scenes):
            self._add_scene_geometry(i, scene_geom)

        # Add trajectory geometries for each frame
        for i, traj_geom in enumerate(self.vis_trajs):
            self._add_trajectory_geometry(i, traj_geom)

        # # # Add camera frustums
        # for i, pose in enumerate(self.slam_poses):
        #     self._add_camera_frustum(i, pose)

    def run(self):
        """Run the web-based visualizer"""
        # Setup GUI controls
        self._setup_gui_controls()
        self._setup_gui_callbacks()

        # Setup 3D scene
        self._setup_scene()

        # Set initial camera position based on first SLAM pose
        if self.vis_param.follow_slam and len(self.slam_poses) > 0:
            self._set_camera_from_slam_pose(self.slam_poses[0], 0)

        print(f"\n=== Web Scene Flow Viewer ===")
        print(f"Viewer: {self.viewer_name}")
        print(f"Frames: {self.vis_param.n_imgs}")
        # print(f"Web interface: http://localhost:{self.server.port}")
        print(f"Controls:")
        print(f"  - Use GUI controls in the web interface")
        print(f"  - Frame slider: Navigate through frames")
        print(f"  - Play/Pause: Animate through frames")
        print(f"  - Toggle checkboxes: Control visibility")
        print(f"  - Reset View: Return to default camera")
        print(f"================================")

        # Keep the server running
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down web viewer...")
            if self.animation_thread and self.animation_thread.is_alive():
                self.stop_animation = True
                self.animation_thread.join()


SceneViewer = edict({"o3d": O3dSceneFlowViewer, "viser": ViserSceneFlowViewer})
