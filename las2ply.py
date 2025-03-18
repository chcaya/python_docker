import pdal
from pyproj import Geod
from affine import Affine
import math
import numpy as np

input_file = 'outputs/landings.las'
output_file = 'outputs/landings.ply'

# landings origin
# ORIGIN_LAT = 45.378541  # Latitude in degrees
# ORIGIN_LONG = -71.942203  # Longitude in degrees
# YAW = 0

# rtabmap_cloud origin
ORIGIN_LAT = 45.3777769  # Latitude in degrees
ORIGIN_LONG = -71.9403259  # Longitude in degrees
YAW = 0

def get_meters_per_degree_at_coord(_coord_long, _coord_lat):
    geod = Geod(ellps="WGS84")
    _, _, meters_per_degree_lon = geod.inv(_coord_long, _coord_lat, _coord_long + 1, _coord_lat)
    _, _, meters_per_degree_lat = geod.inv(_coord_long, _coord_lat, _coord_long, _coord_lat + 1)

    return meters_per_degree_lon, meters_per_degree_lat

def build_affines(_org_x, _org_y, _scale_x, _scale_y, _yaw):
    translation_matrix = Affine.translation(_org_x, _org_y)
    scaling_matrix = Affine.scale(1/_scale_x, 1/_scale_y)
    rotation_matrix = Affine.rotation(math.degrees(_yaw))
    transform = translation_matrix * scaling_matrix * rotation_matrix

    invert_rotation_matrix = Affine.rotation(-math.degrees(_yaw))
    invert_scaling_matrix = Affine.scale(_scale_x, _scale_y)
    invert_translation_matrix = Affine.translation(-_org_x, -_org_y)
    invert_transform = invert_rotation_matrix * invert_scaling_matrix * invert_translation_matrix

    # transform = local to global
    # invert_transform = global to local
    return transform, invert_transform

def affine_to_3d(_affine):
    # Convert the affine matrix to a 4x4 matrix for PDAL
    affine_np = np.array(_affine).reshape(3, 3)
    matrix_3d = np.eye(4)  # Create a 4x4 identity matrix
    matrix_3d[:2, :2] = affine_np[:2, :2]  # Copy rotation and scaling
    matrix_3d[:2, 3] = affine_np[:2, 2]    # Copy translation

    print("\nTransformation Matrix:")
    for row in matrix_3d:
        print(f"|{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}|")

    return matrix_3d

meters_per_degree_lon, meters_per_degree_lat = get_meters_per_degree_at_coord(ORIGIN_LONG, ORIGIN_LAT)

transform, invert_transform = build_affines(ORIGIN_LONG, ORIGIN_LAT, meters_per_degree_lon, meters_per_degree_lat, YAW)

matrix_3d = affine_to_3d(transform)
invert_matrix_3d = affine_to_3d(invert_transform)

pipeline_json = """
[
    {
        "type": "readers.las",
        "filename": "{INPUT}"
    },
    {
        "type": "filters.transformation",
        "matrix": "{MATRIX}"
    },
    {
        "type": "writers.ply",
        "filename": "{OUTPUT}"
    }
]
"""

# Replace the placeholder with the actual matrix
pipeline_json = pipeline_json.replace("{INPUT}", input_file)
pipeline_json = pipeline_json.replace("{OUTPUT}", output_file)
pipeline_json = pipeline_json.replace("{MATRIX}", " ".join(map(str, invert_matrix_3d.flatten())))

# Create the PDAL pipeline
pipeline = pdal.Pipeline(pipeline_json)

# Execute the pipeline
pipeline.execute()



# import open3d as o3d

# # Load the first PLY file
# ply_file_1 = "rtabmap_cloud.ply"
# point_cloud_1 = o3d.io.read_point_cloud(ply_file_1)

# # Load the second PLY file
# ply_file_2 = "landings.ply"
# point_cloud_2 = o3d.io.read_point_cloud(ply_file_2)

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(point_cloud_1)

# for i, point in enumerate(point_cloud_2.points):
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
#     sphere.paint_uniform_color(point_cloud_2.colors[i])
#     sphere.translate(point)
#     vis.add_geometry(sphere)

# vis.run()
# vis.destroy_window()
