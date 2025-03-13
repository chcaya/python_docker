from deepforest import main
from deepforest import get_data
from deepforest import visualize

import tifffile as tif
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
import geopandas as gpd

# # Open the TIFF file
# with rasterio.open('/home/docker/python_docker/images/stac/20240909_bcifairchild_m3e_rgb.cog.tif') as src:
#     # Define the scaling factor
#     # Model 10cm/pixel
#     # Raster 2.14cm/pixel
#     scaling_factor = 10 / 2.14

#     # Calculate new dimensions
#     new_width = int(src.width / scaling_factor)
#     new_height = int(src.height / scaling_factor)

#     # Read the image in chunks and downsample
#     downsampled_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
#     width_size = 1000
#     height_size = 1000
#     for i in range(0, src.height, height_size):
#         for j in range(0, src.width, width_size):
#             window = Window(j, i, min(width_size, src.width - j), min(height_size, src.height - i))
#             chunk = src.read(window=window)
#             # Remove the alpha channel (if present) and move channels to the last dimension
#             if chunk.shape[0] == 4:  # Check if there are 4 channels (RGB + Alpha)
#                 chunk = chunk[:3, :, :]  # Keep only the first 3 channels (RGB)
#             chunk = np.moveaxis(chunk, 0, -1)  # Move channels to the last dimension
#             chunk = cv2.resize(chunk, (int(window.width / scaling_factor), int(window.height / scaling_factor)), interpolation=cv2.INTER_LINEAR)
#             downsampled_img[
#                 int(i / scaling_factor):int((i + window.height) / scaling_factor),
#                 int(j / scaling_factor):int((j + window.width) / scaling_factor),
#                 :
#             ] = chunk

# # Save the downsampled image
# tif.imwrite('/home/docker/python_docker/images/stac/20240909_bcifairchild_m3e_rgb_downsampled.cog.tif', downsampled_img)

# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# Tile pipeline from image
# downsampled_img = tif.imread('/home/docker/python_docker/images/stac/20240909_bcifairchild_m3e_rgb_downsampled.cog.tif')
# results = model.predict_tile(image=downsampled_img, patch_size=800, patch_overlap=0.25)
# results.root_dir = '/home/docker/python_docker/output/'
# results.image_path = '/home/docker/python_docker/images/stac/20240909_bcifairchild_m3e_rgb_downsampled.cog.tif'

# # Tile pipeline from path
# image_path = get_data("/home/docker/python_docker/images/stac/20241125_bci25haplot_m3e_dsm_opaque.cog.tif")
# results = model.predict_tile(image_path, patch_size=800, patch_overlap=0.25)
# visualize.plot_results(results)

# Image pipeline from image
image_path = "/home/docker/python_docker/images/simulation/img_65m.png"
img = cv2.imread(image_path)
# results = model.predict_image(image=img)
results = model.predict_tile(image=img, patch_size=800, patch_overlap=0.25)
results.root_dir = '/home/docker/python_docker/output/'
results.image_path = image_path

# Finding the center of the biggest tree
print('Results type: ' + str(type(results)))
results["area"] = (results["xmax"] - results["xmin"]) * (results["ymax"] - results["ymin"])
results["x"] = (results["xmax"] - results["xmin"])/2.0 + results["xmin"]
results["y"] = (results["ymax"] - results["ymin"])/2.0 + results["ymin"]

max_area_idx = results["area"].idxmax()
center_x = int(results.loc[max_area_idx, "x"])
center_y = int(results.loc[max_area_idx, "y"])

# Output bounding results
print(results)
print(f"Center of the Largest Bounding Box: ({center_x}, {center_y})")

visualize.plot_results(results, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
