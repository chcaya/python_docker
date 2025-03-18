import pdal
from pyproj import Geod
from affine import Affine
import math
import numpy as np
import open3d as o3d

input_file = 'inputs/rtabmap_cloud.ply'

output_file = 'outputs/output_sim.las'

# rtabmap_cloud
ORIGIN_LAT = 45.3777769  # Latitude in degrees
ORIGIN_LONG = -71.9403259  # Longitude in degrees
# YAW = math.radians(92.82) - math.radians(90)
YAW = 0

def get_meters_per_degree_at_coord(_coord_long, _coord_lat):
    # Define the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")

    # Calculate meters per degree of longitude (change in longitude, keep latitude constant)
    _, _, meters_per_degree_lon = geod.inv(_coord_long, _coord_lat, _coord_long + 1, _coord_lat)

    # Calculate meters per degree of latitude (change in latitude, keep longitude constant)
    _, _, meters_per_degree_lat = geod.inv(_coord_long, _coord_lat, _coord_long, _coord_lat + 1)

    return meters_per_degree_lon, meters_per_degree_lat

def affine2pdal(_affine):
    # Convert the affine matrix to a 4x4 matrix for PDAL
    affine_np = np.array(_affine).reshape(3, 3)
    matrix_3d = np.eye(4)  # Create a 4x4 identity matrix
    matrix_3d[:2, :2] = affine_np[:2, :2]  # Copy rotation and scaling
    matrix_3d[:2, 3] = affine_np[:2, 2]    # Copy translation

    print("\nTransformation Matrix:")
    for row in matrix_3d:
        print(f"|{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f}|")

    # Flatten the matrix to a space-separated string for PDAL
    return " ".join(map(str, matrix_3d.flatten()))

meters_per_degree_lon, meters_per_degree_lat = get_meters_per_degree_at_coord(ORIGIN_LONG, ORIGIN_LAT)

# Load the point cloud from the PLY file
point_cloud = o3d.io.read_point_cloud(input_file)

# Get the points as a NumPy array
points = np.asarray(point_cloud.points)

# Compute the minimum and maximum values for X and Y
min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

# Compute the dimensions (width and height)
x_dimension = max_x - min_x
y_dimension = max_y - min_y

# Create the individual matrices
translation_matrix = Affine.translation(ORIGIN_LONG, ORIGIN_LAT)
scaling_matrix = Affine.scale(1/meters_per_degree_lon, 1/meters_per_degree_lat)
rotation_matrix = Affine.rotation(math.degrees(YAW))
transform = translation_matrix * scaling_matrix * rotation_matrix

# Print the results
print(f"X Dimension (Width): {x_dimension:.8f} units")
print(f"Y Dimension (Height): {y_dimension:.8f} units")
print(f"meters_per_degree_lon: {meters_per_degree_lon:.10f}")
print(f"meters_per_degree_lat: {meters_per_degree_lat:.10f}")
print(f"1/meters_per_degree_lon: {1/meters_per_degree_lon:.10f}")
print(f"1/meters_per_degree_lat: {1/meters_per_degree_lat:.10f}")

# Print the individual matrices
print("\nTranslation Matrix:")
print(f"|{translation_matrix.a:.8f}, {translation_matrix.b:.8f}, {translation_matrix.c:.8f}|")
print(f"|{translation_matrix.d:.8f}, {translation_matrix.e:.8f}, {translation_matrix.f:.8f}|")
print(f"|{translation_matrix.g:.8f}, {translation_matrix.h:.8f}, {translation_matrix.i:.8f}|")

print("\nRotation Matrix:")
print(f"|{rotation_matrix.a:.8f}, {rotation_matrix.b:.8f}, {rotation_matrix.c:.8f}|")
print(f"|{rotation_matrix.d:.8f}, {rotation_matrix.e:.8f}, {rotation_matrix.f:.8f}|")
print(f"|{rotation_matrix.g:.8f}, {rotation_matrix.h:.8f}, {rotation_matrix.i:.8f}|")

print("\nScaling Matrix:")
print(f"|{scaling_matrix.a:.8f}, {scaling_matrix.b:.8f}, {scaling_matrix.c:.8f}|")
print(f"|{scaling_matrix.d:.8f}, {scaling_matrix.e:.8f}, {scaling_matrix.f:.8f}|")
print(f"|{scaling_matrix.g:.8f}, {scaling_matrix.h:.8f}, {scaling_matrix.i:.8f}|")

print(f"\nImage transform:")
print(f"|{transform.a:.8f}, {transform.b:.8f}, {transform.c:.8f}|")
print(f"|{transform.d:.8f}, {transform.e:.8f}, {transform.f:.8f}|")
print(f"|{transform.g:.8f}, {transform.h:.8f}, {transform.i:.8f}|")

# Flatten the matrix to a space-separated string for PDAL
pdal_matrix = affine2pdal(transform)

pipeline_json = """
[
    {
        "type": "readers.ply",
        "filename": "{INPUT}"
    },
    {
        "type": "filters.transformation",
        "matrix": "{MATRIX}"
    },
    {
        "type": "writers.las",
        "a_srs": "EPSG:4326",
        "filename": "{OUTPUT}",
        "scale_x":"0.0000001",
        "scale_y":"0.0000001",
        "scale_z":"0.0000001"
    }
]
"""

# Replace the placeholder with the actual matrix
pipeline_json = pipeline_json.replace("{INPUT}", input_file)
pipeline_json = pipeline_json.replace("{OUTPUT}", output_file)
pipeline_json = pipeline_json.replace("{MATRIX}", pdal_matrix)

# Create the PDAL pipeline
pipeline = pdal.Pipeline(pipeline_json)

# Execute the pipeline
pipeline.execute()
