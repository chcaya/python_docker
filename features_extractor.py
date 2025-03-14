import open3d as o3d
import numpy as np

def get_highest_point(_points):
    z_coordinates = _points[:, 2]  # Extract the Z-coordinates
    highest_point_index = np.argmax(z_coordinates)  # Index of the highest point
    return points[highest_point_index]  # Coordinates of the highest point

def get_neighborhood_points(_point, _radius, _points, _normals, _kdtree):
    # Perform a radius search (find all points within a radius of the target point)
    [k, idx, _] = _kdtree.search_radius_vector_3d(_point, _radius)

    # Alternatively, perform a K-nearest neighbors search
    # k = 10  # Number of nearest neighbors
    # [k, idx, _] = _kdtree.search_knn_vector_3d(highest_point, k)

    # Extract the neighborhood points
    neighborhood_points = _points[idx]

    # Compute features for the neighborhood (e.g., normals, curvature)
    neighborhood_normals = np.asarray(_normals)[idx]

    # Visualize the neighborhood
    neighborhood_cloud = o3d.geometry.PointCloud()
    neighborhood_cloud.points = o3d.utility.Vector3dVector(neighborhood_points)
    neighborhood_cloud.normals = o3d.utility.Vector3dVector(neighborhood_normals)

    return neighborhood_cloud

def get_biggest_cluster(_points, _normals, _eps, _min_points):
    # Perform DBSCAN clustering
    # Maximum distance between two points to be considered neighbors
    # Minimum number of points to form a cluster
    labels = np.array(point_cloud.cluster_dbscan(eps=_eps, min_points=_min_points, print_progress=True))

    # Get the number of clusters (ignore noise, which has label -1)
    max_label = labels.max()
    print(f"Number of clusters: {max_label + 1}")

    # Count the number of points in each cluster
    cluster_sizes = [np.sum(labels == i) for i in range(max_label + 1)]

    # Find the biggest cluster
    biggest_cluster_idx = np.argmax(cluster_sizes)
    print(f"Biggest cluster index: {biggest_cluster_idx}, Size: {cluster_sizes[biggest_cluster_idx]}")

    # Extract a specific cluster (e.g., cluster 0)
    biggest_cluster_mask = labels == biggest_cluster_idx
    biggest_cluster_points = _points[biggest_cluster_mask]
    biggest_cluster_normals = _normals[biggest_cluster_mask]

    # Create a new point cloud for the cluster
    biggest_cluster_cloud = o3d.geometry.PointCloud()
    biggest_cluster_cloud.points = o3d.utility.Vector3dVector(biggest_cluster_points)
    biggest_cluster_cloud.normals = o3d.utility.Vector3dVector(biggest_cluster_normals)

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

points = np.asarray(point_cloud.points)
normals = np.asarray(point_cloud.normals)

highest_point = get_highest_point(points)

# Downsample using a voxel grid
voxel_size = 0.2  # Size of the voxel (adjust based on your data)
downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)

points = np.asarray(downsampled_cloud.points)
normals = np.asarray(downsampled_cloud.normals)
print(downsampled_cloud)
print(len(points))
print(len(normals))


kdtree = o3d.geometry.KDTreeFlann(downsampled_cloud)

radius = 1.5
neighborhood_cloud = get_neighborhood_points(highest_point, radius, points, normals, kdtree)
red = [1, 0, 0]
neighborhood_cloud.paint_uniform_color(red)

# biggest_cluster_cloud = get_biggest_cluster(points, normals, 0.5, 100)
# blue = [0, 0, 1]
# biggest_cluster_cloud.paint_uniform_color(blue)

curvature = compute_curvature(neighborhood_cloud.normals)
density = compute_density(neighborhood_cloud.points, radius)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)
orange = [1, 0.5, 0]
highlight_point(vis, highest_point, orange)
vis.add_geometry(neighborhood_cloud)

vis.run()
vis.destroy_window()
