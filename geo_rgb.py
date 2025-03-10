import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt

import math
from pyproj import Geod
from affine import Affine


def compute_img_size(_fov, _height):
    return 2*_height*math.tan(math.radians(_fov/2.0))

HEIGHT = 65

CAM_FOV_H = 87
CAM_FOV_V = 58

CENTER_LAT = 45.0  # Latitude in degrees
CENTER_LONG = -72.0  # Longitude in degrees
YAW = math.radians(15)

image_width_meters = compute_img_size(CAM_FOV_H, HEIGHT)
image_height_meters = compute_img_size(CAM_FOV_V, HEIGHT)

# Define the WGS84 ellipsoid
geod = Geod(ellps="WGS84")

# Calculate meters per degree of longitude (change in longitude, keep latitude constant)
_, _, meters_per_degree_lon = geod.inv(CENTER_LONG, CENTER_LAT, CENTER_LONG + 1, CENTER_LAT)

# Calculate meters per degree of latitude (change in latitude, keep longitude constant)
_, _, meters_per_degree_lat = geod.inv(CENTER_LONG, CENTER_LAT, CENTER_LONG, CENTER_LAT + 1)

print(f"Values: {image_width_meters}, {meters_per_degree_lon}, {image_height_meters}, {meters_per_degree_lat}")

# Calculate the image dimensions in degrees
image_width_degrees = image_width_meters / meters_per_degree_lon
image_height_degrees = image_height_meters / meters_per_degree_lat

print(f"Meters per degree of longitude at latitude {CENTER_LAT}: {meters_per_degree_lon}")
print(f"Meters per degree of latitude at latitude {CENTER_LAT}: {meters_per_degree_lat}")

# Load the PNG image as a numpy array
png_image = image.imread('img.png')

image_height_pixels = png_image.shape[0]
image_width_pixels = png_image.shape[1]

# Define the local coordinates of the corners (relative to the center)
corners_local = [
    (-image_width_meters / 2.0, image_height_meters / 2.0),  # Top-left
    (image_width_meters / 2.0, image_height_meters / 2.0),   # Top-right
    (-image_width_meters / 2.0, -image_height_meters / 2.0), # Bottom-left
    (image_width_meters / 2.0, -image_height_meters / 2.0),  # Bottom-right
]

# Rotate the corners around the center
corners_rotated = [
    (
        x * math.cos(YAW) - y * math.sin(YAW),  # Rotated x
        x * math.sin(YAW) + y * math.cos(YAW),  # Rotated y
    )
    for x, y in corners_local
]

# Convert the rotated local coordinates to geographic coordinates
corners_geo = [
    (CENTER_LONG + (x / meters_per_degree_lon), CENTER_LAT + (y / meters_per_degree_lat))
    for x, y in corners_rotated
]

print("Geo corners: " + str(corners_geo))

# Find the top-left corner in geographic coordinates
top_left_lon = corners_geo[0][0]
top_left_lat = corners_geo[0][1]

# top_left_lon = (-image_width_meters / 2.0) / meters_per_degree_lon
# top_left_lat = (image_height_meters / 2.0) / meters_per_degree_lat

print(f"Values: {image_width_degrees}, {image_width_pixels}, {image_height_degrees}, {image_height_pixels}")

# Calculate the pixel size in degrees
pixel_size_x_degrees = image_width_degrees / image_width_pixels
pixel_size_y_degrees = -image_height_degrees / image_height_pixels  # Negative because y-axis is flipped

# Debugging: Print the pixel sizes
print(f"Pixel size (X): {pixel_size_x_degrees:.8f} degrees")
print(f"Pixel size (Y): {pixel_size_y_degrees:.8f} degrees")

print(f"Values: {top_left_lon}, {top_left_lat}, {YAW}, {pixel_size_x_degrees}, {pixel_size_y_degrees}")
# # Create the affine transformation matrix with rotation
# transform = (
#     Affine.translation(top_left_lon, top_left_lat) *
#     Affine.rotation(YAW) *
#     Affine.scale(pixel_size_x_degrees, pixel_size_y_degrees)
# )

# Create the individual matrices
translation_matrix = Affine.translation(top_left_lon, top_left_lat)
rotation_matrix = Affine.rotation(math.degrees(YAW))
scaling_matrix = Affine.scale(pixel_size_x_degrees, pixel_size_y_degrees)

# Print the individual matrices
print("Translation Matrix:")
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

# Combine the matrices
transform = translation_matrix * rotation_matrix * scaling_matrix

print(f"Image transform: |{transform.a:.8f}, {transform.b:.8f}, {transform.c:.8f}|")
print(f"                 |{transform.d:.8f}, {transform.e:.8f}, {transform.f:.8f}|")
print(f"                 |{transform.g:.8f}, {transform.h:.8f}, {transform.i:.8f}|")

# Define the CRS (e.g., WGS84)
crs = 'EPSG:4326'
# crs = 'EPSG:32618'

# Save the geo-referenced image as a GeoTIFF
with rasterio.open(
    'output_georeferenced.tif',
    'w',
    driver='GTiff',
    height=png_image.shape[0],
    width=png_image.shape[1],
    count=3,  # Number of bands (3 for RGB)
    dtype=png_image.dtype,
    crs=crs,
    transform=transform,
) as dst:
    # Write the RGB bands to the GeoTIFF
    for band in range(3):
        dst.write(png_image[:, :, band], band + 1)


# # Open the GeoTIFF file
# with rasterio.open('output_georeferenced.tif') as src:
#     # Read the first band (or all bands for RGB images)
#     image =  src.read([1, 2, 3]) # Change to `src.read()` for multi-band images

#     # Plot the image
#     plt.figure(figsize=(10, 10))
#     show(image)
#     plt.colorbar(label='Pixel Value')
#     plt.title('GeoTIFF Image')
#     plt.show()

# # Define the source CRS (WGS84)
# src_crs = 'EPSG:4326'

# # Define the target CRS (e.g., UTM Zone 18N)
# dst_crs = 'EPSG:32618'

# # Calculate the transform and dimensions for the reprojected image
# transform, width, height = calculate_default_transform(
#     src_crs, dst_crs, image_width_pixels, image_height_pixels,
#     *rasterio.transform.array_bounds(image_height_pixels, image_width_pixels, transform)
# )

# # Print the calculated transform and dimensions
# print(f"Reprojected image dimensions: {width}x{height}")
# print(f"Reprojected transform: {transform}")

# # Create a new array for the reprojected image
# reprojected_image = np.zeros((3, height, width), dtype=png_image.dtype)

# # Reproject the image
# reproject(
#     source=png_image.transpose(2, 0, 1),  # Move bands to the first dimension
#     destination=reprojected_image,
#     src_transform=transform,
#     src_crs=src_crs,
#     dst_transform=transform,
#     dst_crs=dst_crs,
#     resampling=Resampling.nearest
# )

# # Verify the reprojected image
# print(f"Reprojected image min: {np.min(reprojected_image)}")
# print(f"Reprojected image max: {np.max(reprojected_image)}")

# # Save the reprojected image as a GeoTIFF
# with rasterio.open(
#     'output_reprojected.tif',
#     'w',
#     driver='GTiff',
#     height=height,
#     width=width,
#     count=3,
#     dtype=reprojected_image.dtype,
#     crs=dst_crs,
#     transform=transform,
# ) as dst:
#     for band in range(3):
#         dst.write(reprojected_image[band], band + 1)