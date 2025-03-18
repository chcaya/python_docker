#include <iostream>
#include <pcl/io/ply_io.h> // For loading PLY files
#include <pcl/point_types.h> // For point cloud types
#include <pcl/visualization/pcl_visualizer.h> // For visualization (optional)

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPly(std::string _filePath){
    // Create a point cloud object for XYZRGB points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Load the .ply file
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(_filePath, *cloud) == -1) {
        std::cerr << "Error: Could not load .ply file: " << _filePath << std::endl;
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr();
    }

    // Print the number of points in the cloud
    std::cout << "Loaded " << cloud->width * cloud->height << " points from " << _filePath << std::endl;

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

    return cloud;
}

void visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud){
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (_cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    // https://github.com/PointCloudLibrary/pcl/issues/5237#issuecomment-1114255056
    // spin() avoids crash
    viewer->spin();
}

int main() {
    std::string ply_file_path = "../inputs/rtabmap_cloud.ply";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = loadPly(ply_file_path);

    // Landing point [15.50081099  3.76794873 12.62245941]

    visualize(cloud);
    return 0;
}
