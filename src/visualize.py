import numpy as np
import open3d as o3d


class Visualize3D:
    def __init__(self, locs, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(locs)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self.trajectory_points = []
        self.geometries_to_draw = []
        self.geometries_to_draw.append(pcd)

        # o3d.io.write_point_cloud("output.ply", pcd)
        # o3d.visualization.draw_geometries([pcd, camera])

    def add_pose(self, T):
        T_inv = np.linalg.inv(T)
        frustum_lines, frustum_base = self.create_camera_frustum()
        frustum_lines.transform(T_inv)
        frustum_base.transform(T_inv)

        self.add_trajectory(T_inv)
        self.geometries_to_draw.append(frustum_lines)
        self.geometries_to_draw.append(frustum_base)
        return

    def create_camera_frustum(self, size=0.5):
        # Define the pyramid's 5 vertices
        # Apex of the pyramid

        # Base of the pyramid (square) - assuming camera looks along +Z and Y is up
        half_size = size / 2.0
        points = [
            [0,0,0],
            [ half_size,  half_size, size],
            [-half_size,  half_size, size],
            [-half_size, -half_size, size],
            [ half_size, -half_size, size]
        ]

        # Create lines to represent the pyramid edges
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4]
        ]

        # Convert lines to open3d LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Create a triangle mesh for the blue bottom
        # Split the quadrilateral into two triangles
        triangles = [
            [1, 2, 3],
            [1, 3, 0],
            [3, 2, 1],
            [0, 3, 1]
        ]

        base = [
            [ half_size,  half_size, size],
            [-half_size,  half_size, size],
            [-half_size, -half_size, size],
            [ half_size, -half_size, size]
        ]

        triangle_mesh = o3d.geometry.TriangleMesh()
        triangle_mesh.vertices = o3d.utility.Vector3dVector(base)
        triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        triangle_mesh.vertex_colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in base])  # Blue color
        return line_set, triangle_mesh

    def add_trajectory(self, T_inv):
        origin = np.array([0, 0, 0, 1])
        self.trajectory_points.append((T_inv @ origin)[:3])
        return

    def construct_trajectory(self):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.trajectory_points)
        lines = np.array([[i, i+1]for i in range(len(self.trajectory_points)-1)])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (lines.shape[0], 1)))  # RGB values for green
        self.geometries_to_draw.append(line_set)
        return

    def draw_geometry(self):
        self.construct_trajectory()
        o3d.visualization.draw_geometries(self.geometries_to_draw)
        return
