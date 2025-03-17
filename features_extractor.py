import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def get_highest_point(_points):
    z_coordinates = _points[:, 2]  # Extract the Z-coordinates
    highest_point_index = np.argmax(z_coordinates)  # Index of the highest point
    return points[highest_point_index]  # Coordinates of the highest point

def get_neighborhood_points(_pcd, _kdtree, _point, _radius):
    # Perform a radius search (find all points within a radius of the target point)
    [k, idx, _] = _kdtree.search_radius_vector_3d(_point, _radius)

    # Alternatively, perform a K-nearest neighbors search
    # k = 10  # Number of nearest neighbors
    # [k, idx, _] = _kdtree.search_knn_vector_3d(highest_point, k)

    # Extract the neighborhood points
    neighborhood_points = _pcd.points[idx]

    # Compute features for the neighborhood (e.g., normals, curvature)
    neighborhood_normals = np.asarray(_pcd.normals)[idx]

    # Visualize the neighborhood
    neighborhood_cloud = o3d.geometry.PointCloud()
    neighborhood_cloud.points = o3d.utility.Vector3dVector(neighborhood_points)
    neighborhood_cloud.normals = o3d.utility.Vector3dVector(neighborhood_normals)

    return neighborhood_cloud

def get_biggest_cluster(_pcd, _kdtree, _radius, _min_cluster_size):
    """
    Find the biggest cluster in a point cloud using Euclidean clustering.

    Args:
        _pcd (o3d.geometry.PointCloud): The input point cloud.
        _kdtree
        _radius (float): The search radius for clustering.
        _min_cluster_size (int): Minimum number of points in a cluster.

    Returns:
        o3d.geometry.PointCloud: The biggest cluster as a point cloud.
    """
    # Convert point cloud to numpy array
    points = np.asarray(_pcd.points)

    # Initialize variables
    clusters = []  # List to store clusters
    visited = set()  # Set to keep track of visited points

    # Iterate through all points
    for i in range(len(points)):
        if i in visited:
            continue

        # Start a new cluster
        cluster = []
        stack = [i]  # Use a stack for region growing

        while stack:
            point_idx = stack.pop()
            if point_idx in visited:
                continue

            visited.add(point_idx)
            cluster.append(point_idx)

            # Find neighbors within the search radius
            # https://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html
            [k, idx, _] = _kdtree.search_radius_vector_3d(_pcd.points[point_idx], _radius)
            stack.extend(idx)

        # Add cluster if it meets the minimum size requirement
        if len(cluster) >= _min_cluster_size:
            clusters.append(cluster)

    # Find the biggest cluster
    if not clusters:
        raise ValueError("No clusters found. Adjust the radius or min_cluster_size.")

    biggest_cluster = max(clusters, key=len)

    # Create a new point cloud for the biggest cluster
    biggest_cluster_cloud = o3d.geometry.PointCloud()
    biggest_cluster_cloud.points = o3d.utility.Vector3dVector(points[biggest_cluster])

    return biggest_cluster_cloud

def compute_curvature(_normals):
    curvature = np.var(_normals, axis=0)
    print("Curvature:", curvature)
    return curvature

def compute_density(_points, _radius):
    volume = (4/3) * np.pi * (_radius ** 3)
    density = len(_points) / volume
    print("Density:", density)
    return density

def highlight_point(_vis, _point, _color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color(_color)
    sphere.translate(_point)
    _vis.add_geometry(sphere)



point_cloud = o3d.io.read_point_cloud("rtabmap_cloud.ply")

if not point_cloud.has_normals():
    print("Computing normals")
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

print(point_cloud)

highest_point = get_highest_point(point_cloud.points)

# Downsample using a voxel grid
voxel_size = 0.2  # Size of the voxel (adjust based on your data)
downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)

print(downsampled_cloud)

ds_kdtree = o3d.geometry.KDTreeFlann(downsampled_cloud)

radius = 1.5
neighborhood_cloud = get_neighborhood_points(downsampled_cloud, ds_kdtree, highest_point, radius)
red = [1, 0, 0]
neighborhood_cloud.paint_uniform_color(red)

biggest_cluster_cloud = get_biggest_cluster(downsampled_cloud, ds_kdtree, 0.5, 100)
blue = [0, 0, 1]
biggest_cluster_cloud.paint_uniform_color(blue)

curvature = compute_curvature(neighborhood_cloud.normals)
density = compute_density(neighborhood_cloud.points, radius)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)
orange = [1, 0.5, 0]
highlight_point(vis, highest_point, orange)
vis.add_geometry(neighborhood_cloud)
vis.add_geometry(biggest_cluster_cloud)

vis.run()
vis.destroy_window()
