#include <iostream>
#include <pcl/io/ply_io.h> // For loading PLY files
#include <pcl/point_types.h> // For point cloud types
#include <pcl/visualization/pcl_visualizer.h> // For visualization (optional)

void visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud){
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (_cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    viewer->spin();
}

int main() {
    // Hardcode the path to the .ply file
    std::string ply_file_path = "../inputs/rtabmap_cloud.ply"; // Replace with your file path

    // Create a point cloud object for XYZRGB points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Load the .ply file
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(ply_file_path, *cloud) == -1) {
        std::cerr << "Error: Could not load .ply file: " << ply_file_path << std::endl;
        return -1;
    }

    // Print the number of points in the cloud
    std::cout << "Loaded " << cloud->width * cloud->height << " points from " << ply_file_path << std::endl;

    // Print the first 5 points with their RGB values (optional)
    for (size_t i = 0; i < 5 && i < cloud->points.size(); ++i) {
        const auto& point = cloud->points[i];
        std::cout << "Point " << i << ": "
                  << "X: " << point.x << ", "
                  << "Y: " << point.y << ", "
                  << "Z: " << point.z << ", "
                  << "R: " << static_cast<int>(point.r) << ", "
                  << "G: " << static_cast<int>(point.g) << ", "
                  << "B: " << static_cast<int>(point.b) << std::endl;
    }

    visualize(cloud);

    return 0;
}
