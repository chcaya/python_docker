import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("inputs/rtabmap_cloud.ply")

print(pcd)

points = np.asarray(pcd.points)

# Compute the center (mean of all points)
center = np.mean(points, axis=0)

# Highlight points (large)
mean = center  # Mean of the normal distribution
std_dev = 1  # Standard deviation of the normal distribution
success_landing_points = np.random.normal(mean, std_dev, size=(4, 3))
failed_landing_points = np.random.normal(mean, std_dev, size=(3, 3))

# success_landing_points = np.random.rand(10, 3)
# failed_landing_points = np.random.rand(10, 3)

# Create highlight point cloud
success_landing_pcd = o3d.geometry.PointCloud()
success_landing_pcd.points = o3d.utility.Vector3dVector(success_landing_points)

failed_landing_pcd = o3d.geometry.PointCloud()
failed_landing_pcd.points = o3d.utility.Vector3dVector(failed_landing_points)

# Start visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add base points (small size)
vis.add_geometry(pcd)

# Add highlighted points (larger size)
vis.add_geometry(success_landing_pcd)
vis.add_geometry(failed_landing_pcd)

# Increase size for highlights by creating a sphere at each point
for point in success_landing_points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color([0, 1, 0])
    sphere.translate(point)
    vis.add_geometry(sphere)

for point in failed_landing_points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(point)
    vis.add_geometry(sphere)

# Run visualization
vis.run()
vis.destroy_window()
