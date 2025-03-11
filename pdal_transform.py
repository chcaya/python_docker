import pdal

# Define the PDAL pipeline
pipeline = pdal.Pipeline(
    """
    [
        "rtabmap_cloud.ply",
        {
            "type": "writers.las",
            "filename": "output.las"
        }
    ]
    """
)

# Execute the pipeline
pipeline.execute()
