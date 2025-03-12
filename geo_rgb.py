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

def get_meters_per_degree_at_coord(_coord_long, _coord_lat):
    # Define the WGS84 ellipsoid
    geod = Geod(ellps="WGS84")

    # Calculate meters per degree of longitude (change in longitude, keep latitude constant)
    _, _, meters_per_degree_lon = geod.inv(_coord_long, _coord_lat, _coord_long + 1, _coord_lat)

    # Calculate meters per degree of latitude (change in latitude, keep longitude constant)
    _, _, meters_per_degree_lat = geod.inv(_coord_long, _coord_lat, _coord_long, _coord_lat + 1)

    return meters_per_degree_lon, meters_per_degree_lat

HEIGHT = 65

CAM_FOV_H = 87
CAM_FOV_V = 58

CENTER_LAT = 45.0  # Latitude in degrees
CENTER_LONG = -72.0  # Longitude in degrees
YAW = math.radians(15)

image_width_meters = compute_img_size(CAM_FOV_H, HEIGHT)
image_height_meters = compute_img_size(CAM_FOV_V, HEIGHT)

meters_per_degree_lon, meters_per_degree_lat = get_meters_per_degree_at_coord(CENTER_LONG, CENTER_LAT)

print(f"Image Width (meters): {image_width_meters:.2f}")
print(f"Meters per Degree Longitude: {meters_per_degree_lon:.2f}")
print(f"Image Height (meters): {image_height_meters:.2f}")
print(f"Meters per Degree Latitude: {meters_per_degree_lat:.2f}")

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

print(f"Image Width (degrees): {image_width_degrees:.8f}")
print(f"Image Width (pixels): {image_width_pixels:.8f}")
print(f"Image Height (degrees): {image_height_degrees:.8f}")
print(f"Image Height (pixels): {image_height_pixels:.8f}")

# Calculate the pixel size in degrees
pixel_size_x_degrees = image_width_degrees / image_width_pixels
pixel_size_y_degrees = -image_height_degrees / image_height_pixels

print(f"Top-Left Longitude: {top_left_lon:.8f}")
print(f"Top-Left Latitude: {top_left_lat:.8f}")
print(f"YAW: {YAW:.8f}")
print(f"Pixel Size (X, degrees): {pixel_size_x_degrees:.8f}")
print(f"Pixel Size (Y, degrees): {pixel_size_y_degrees:.8f}")

# Create the individual matrices
translation_matrix = Affine.translation(top_left_lon, top_left_lat)
rotation_matrix = Affine.rotation(-math.degrees(YAW)) # Cancels out the flipped Y axis (Mirrored image)
scaling_matrix = Affine.scale(pixel_size_x_degrees, pixel_size_y_degrees)

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

# Combine the matrices
transform = translation_matrix * scaling_matrix * rotation_matrix

print(f"\nImage transform:")
print(f"|{transform.a:.8f}, {transform.b:.8f}, {transform.c:.8f}|")
print(f"|{transform.d:.8f}, {transform.e:.8f}, {transform.f:.8f}|")
print(f"|{transform.g:.8f}, {transform.h:.8f}, {transform.i:.8f}|")

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
