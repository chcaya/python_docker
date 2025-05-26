import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from pyproj import Geod
from affine import Affine


def get_local_coord(_org_x, _org_y, _coord_x, _coord_y):
    geod = Geod(ellps="WGS84")
    _, _, meters_per_degree_lon = geod.inv(_org_x, _org_y, _org_x + 1, _org_y)
    _, _, meters_per_degree_lat = geod.inv(_org_x, _org_y, _org_x, _org_y + 1)

    scale_x = meters_per_degree_lon
    scale_y = meters_per_degree_lat

    scaling_matrix = Affine.scale(scale_x, scale_y)
    translation_matrix = Affine.translation(-_org_x, -_org_y)
    transform = scaling_matrix * translation_matrix

    local_x, local_y = transform * (_coord_x, _coord_y)

    return np.array([local_x, local_y, 0])

def get_highest_point(_pcd):
    points = np.asarray(_pcd.points)
    z_coordinates = points[:, 2]  # Extract the Z-coordinates
    highest_point_index = np.argmax(z_coordinates)  # Index of the highest point
    return points[highest_point_index]  # Coordinates of the highest point

def get_neighborhood_points(_pcd, _kdtree, _point, _radius):
    # Perform a radius search (find all points within a radius of the target point)
    [k, idx, _] = _kdtree.search_radius_vector_3d(_point, _radius)

    # Alternatively, perform a K-nearest neighbors search
    # k = 10  # Number of nearest neighbors
    # [k, idx, _] = _kdtree.search_knn_vector_3d(highest_point, k)

    # Extract the neighborhood points
    neighborhood_points = np.asarray(_pcd.points)[idx]

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

# def mls_smoothing_3d(_pcd, _kdtree, _search_radius=0.2, _polynomial_order=2):
#     """
#     Smooth a point cloud using 3D Moving Least Squares (MLS).

#     Args:
#         _pcd (open3d.geometry.PointCloud): The input point cloud.
#         _search_radius (float): Radius for neighborhood search.
#         _polynomial_order (int): Order of the polynomial to fit.

#     Returns:
#         open3d.geometry.PointCloud: The smoothed point cloud.
#     """
#     points = np.asarray(_pcd.points)
#     smoothed_points = np.zeros_like(points)

#     for i, point in enumerate(points):
#         # Find neighbors within the search radius
#         [k, idx, _] = _kdtree.search_radius_vector_3d(point, _search_radius)
#         neighbors = points[idx]

#         if len(neighbors) < (_polynomial_order + 1) * (_polynomial_order + 2) // 2:
#             # Not enough neighbors to fit a polynomial
#             smoothed_points[i] = point
#             continue

#         # Construct the design matrix A for 3D polynomial fitting
#         A = []
#         for neighbor in neighbors:
#             x, y, z = neighbor
#             row = []
#             for nx in range(_polynomial_order + 1):
#                 for ny in range(_polynomial_order + 1 - nx):
#                     for nz in range(_polynomial_order + 1 - nx - ny):
#                         row.append((x**nx) * (y**ny) * (z**nz))
#             A.append(row)
#         A = np.array(A)

#         # Construct the target vectors (x, y, z)
#         b_x = neighbors[:, 0]
#         b_y = neighbors[:, 1]
#         b_z = neighbors[:, 2]

#         # Solve the least squares problem for each coordinate
#         coeffs_x = np.linalg.lstsq(A, b_x, rcond=None)[0]
#         coeffs_y = np.linalg.lstsq(A, b_y, rcond=None)[0]
#         coeffs_z = np.linalg.lstsq(A, b_z, rcond=None)[0]

#         # Project the point onto the fitted surface
#         x, y, z = point
#         indices = generate_indices(_polynomial_order)
#         smoothed_x = sum(coeffs_x[n] * (x**nx) * (y**ny) * (z**nz) for n, (nx, ny, nz) in enumerate(indices))
#         smoothed_y = sum(coeffs_y[n] * (x**nx) * (y**ny) * (z**nz) for n, (nx, ny, nz) in enumerate(indices))
#         smoothed_z = sum(coeffs_z[n] * (x**nx) * (y**ny) * (z**nz) for n, (nx, ny, nz) in enumerate(indices))

#         smoothed_points[i] = [smoothed_x, smoothed_y, smoothed_z]

#     # Create a new point cloud with the smoothed points
#     smoothed_pcd = o3d.geometry.PointCloud()
#     smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
#     return smoothed_pcd

# def generate_indices(_polynomial_order):
#     """
#     Generate indices for the polynomial terms.
#     """
#     indices = []
#     for nx in range(_polynomial_order + 1):
#         for ny in range(_polynomial_order + 1 - nx):
#             for nz in range(_polynomial_order + 1 - nx - ny):
#                 indices.append((nx, ny, nz))
#     return indices

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


home_latitude=45.3777769
home_longitude=-71.9403259
landing_latitude=45.3778108
landing_longitude=-71.940128

point_cloud = o3d.io.read_point_cloud("inputs/rtabmap_cloud.ply")

if not point_cloud.has_normals():
    print("Computing normals")
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

print(point_cloud)

center_point = get_local_coord(home_longitude, home_latitude, landing_longitude, landing_latitude)
highest_point = get_highest_point(point_cloud)

# Assign the z value of the highest point to the center point
center_point[2] = highest_point[2]  # Update the z coordinate
print(center_point)

# Downsample using a voxel grid
voxel_size = 0.1  # Size of the voxel (adjust based on your data)
downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)

print(downsampled_cloud)

ds_kdtree = o3d.geometry.KDTreeFlann(downsampled_cloud)

radius = 1.5
neighborhood_cloud = get_neighborhood_points(downsampled_cloud, ds_kdtree, center_point, radius)
red = [1, 0, 0]
neighborhood_cloud.paint_uniform_color(red)

biggest_cluster_cloud = get_biggest_cluster(downsampled_cloud, ds_kdtree, 0.5, 100)
blue = [0, 0, 1]
biggest_cluster_cloud.paint_uniform_color(blue)

smooth_radius = 0.3
bc_kdtree = o3d.geometry.KDTreeFlann(biggest_cluster_cloud)
# smoothed_pcd = mls_smoothing_3d(biggest_cluster_cloud, bc_kdtree, smooth_radius, 2)
# orange = [1, 0.5, 0]
# smoothed_pcd.paint_uniform_color(orange)

# # Estimate normals for visualization (optional)
# smoothed_pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2*smooth_radius, max_nn=30)
# )

# curvature = compute_curvature(smoothed_pcd.normals)
density = compute_density(neighborhood_cloud.points, radius)

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(point_cloud)
orange = [1, 0.5, 0]
highlight_point(vis, highest_point, orange)
yellow = [1, 1, 0]
highlight_point(vis, center_point, yellow)
vis.add_geometry(neighborhood_cloud)
vis.add_geometry(biggest_cluster_cloud)
# vis.add_geometry(smoothed_pcd)

vis.run()
vis.destroy_window()
