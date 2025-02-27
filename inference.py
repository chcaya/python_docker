from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results

# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# image_path = get_data("OSBS_029.png")
image_path = get_data("/home/docker/python_docker/images/simulation/scaned_img_65m.png")
# img = model.predict_image(path=image_path)
# raster = model.predict_tile(image_path, patch_size=800, patch_overlap=0.25)
# plot_results(raster)

boxes = model.predict_image(path=image_path, return_plot=False)

boxes["area"] = (boxes["xmax"] - boxes["xmin"]) * (boxes["ymax"] - boxes["ymin"])
boxes["x"] = (boxes["xmax"] - boxes["xmin"])/2.0 + boxes["xmin"]
boxes["y"] = (boxes["ymax"] - boxes["ymin"])/2.0 + boxes["ymin"]

max_area_idx = boxes["area"].idxmax()
center_x = int(boxes.loc[max_area_idx, "x"])
center_y = int(boxes.loc[max_area_idx, "y"])

# Output bounding boxes
print(boxes)
print(f"Center of the Largest Bounding Box: ({center_x}, {center_y})")
