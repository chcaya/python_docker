#include <iostream>
#include <pcl/io/ply_io.h> // For loading PLY files
#include <pcl/point_types.h> // For point cloud types

#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>

#include <pcl/visualization/pcl_visualizer.h> // For visualization (optional)


const int N_NEIGHBORS_SEARCH = 4;

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

template <typename PointT>
void extractPoints(
    const pcl::PointCloud<PointT>& _ogCloud,
    pcl::PointCloud<PointT>& _outputCloud,
    const pcl::PointIndices& _indices,
    bool _isExtractingOutliers
){
    pcl::copyPointCloud(_ogCloud, _outputCloud);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(_ogCloud.makeShared());
    extract.setNegative(_isExtractingOutliers);
    extract.setIndices(std::make_shared<pcl::PointIndices>(_indices));
    extract.filter(_outputCloud);
}

pcl::PointIndices removeNaNFromNormalCloud(pcl::PointCloud<pcl::PointNormal>& _normalsCloud){
    pcl::PointCloud<pcl::PointNormal> tempCloud;
    tempCloud.points.reserve(_normalsCloud.points.size());

    // Create a PointIndices object to store the removed indices
    pcl::PointIndices removedIndices;
    removedIndices.indices.reserve(_normalsCloud.points.size());

    int i = 0;
    for (const auto& normal : _normalsCloud.points) {
        if (!std::isnan(normal.normal_x) && !std::isnan(normal.normal_y) && !std::isnan(normal.normal_z))
        {
            tempCloud.emplace_back(normal);
        }
        else
        {
            // Add index of the point with NaN value to removedIndices
            removedIndices.indices.emplace_back(i);
        }
        ++i;
    }

    pcl::copyPointCloud(tempCloud, _normalsCloud);
    return removedIndices;
}

pcl::PointIndices computeNormalsPointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud,
    pcl::PointCloud<pcl::PointNormal>& _normalsCloud,
    pcl::search::KdTree<pcl::PointXYZRGB>& _kdTree,
    const pcl::PointXYZRGB& _viewPoint,
    const int _searchNeighbors
){
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
    ne.setInputCloud(_pointCloud.makeShared());
    ne.setViewPoint(_viewPoint.x, _viewPoint.y, _viewPoint.z);
    ne.setSearchMethod(std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>(_kdTree));
    ne.setKSearch(_searchNeighbors);
    ne.compute(_normalsCloud);

    pcl::PointIndices removedIdx = removeNaNFromNormalCloud(_normalsCloud);
    extractPoints(_pointCloud, _pointCloud, removedIdx, true);

    return removedIdx;
}

pcl::PointCloud<pcl::PointNormal> extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud,
    const pcl::PointXYZRGB& _centroid
){
    pcl::PointCloud<pcl::PointNormal> normalsCloud;
    pcl::search::KdTree<pcl::PointXYZRGB> kdTree;
    kdTree.setInputCloud(_pointCloud.makeShared());
    computeNormalsPointCloud(_pointCloud, normalsCloud, kdTree, _centroid, N_NEIGHBORS_SEARCH);
    return normalsCloud;
}

pcl::PointXYZRGB getHighestPoint(pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud){
    // Initialize the highest point with the first point in the cloud
    pcl::PointXYZRGB highestPoint = _pointCloud.points[0];

    // Iterate through the cloud to find the highest point
    for (const auto& point : _pointCloud.points) {
        if (point.z > highestPoint.z) {
            highestPoint = point;
        }
    }

    return highestPoint;
}

void downSamplePointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud,
    const float _leafSize
){
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(_pointCloud.makeShared());
    sor.setLeafSize(_leafSize, _leafSize, _leafSize);
    sor.filter(_pointCloud);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractNeighborPC(
    const pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud,
    const pcl::search::KdTree<pcl::PointXYZRGB> _kdtree,
    const pcl::PointXYZRGB _center,
    const float _radius
){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr neighborsPC(new pcl::PointCloud<pcl::PointXYZRGB>);
    neighborsPC->points.reserve(_pointCloud.points.size());

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    if(_kdtree.radiusSearch(_center, _radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i){
            neighborsPC->points.emplace_back(_pointCloud.points[pointIdxRadiusSearch[i]]);
        }
    }

    return neighborsPC;
}

pcl::PointIndices extractBiggestCluster(pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud, const float _threshold, const int _minPoints)
{
    // Cluster extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(_threshold); // Adjust tolerance (distance threshold)
    ec.setMinClusterSize(_minPoints); // Adjust minimum points per cluster
    ec.setMaxClusterSize(std::numeric_limits<int>::max()); // No upper limit
    ec.setInputCloud(_pointCloud.makeShared());
    ec.extract(cluster_indices);

    // Find the largest cluster
    int largest_cluster_index = -1;
    int largest_cluster_size = 0;
    for (size_t i = 0; i < cluster_indices.size(); ++i)
    {
        int cluster_size = cluster_indices[i].indices.size();
        if (cluster_size > largest_cluster_size)
        {
            largest_cluster_index = i;
            largest_cluster_size = cluster_size;
        }
    }

    pcl::PointIndices inliers;
    if (largest_cluster_index != -1)
    {
        inliers = cluster_indices[largest_cluster_index];
        extractPoints(_pointCloud, _pointCloud, inliers, false);
        std::cout << "The point cloud has " << std::to_string(cluster_indices.size()) << " clusters." << std::endl;
    }
    else
    {
        std::cout << "No clusters found in the point cloud." << std::endl;
    }

    return inliers;
}

void smoothPC(pcl::PointCloud<pcl::PointXYZRGB>& _pointCloud, const float _searchRadius)
{
    // Output has the PointNormal type in order to store the normals calculated by MLS
    pcl::PointCloud<pcl::PointNormal> mls_points;

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;
    
    mls.setComputeNormals(true);

    pcl::search::KdTree<pcl::PointXYZRGB> pointTree;
    // Set parameters
    mls.setInputCloud(_pointCloud.makeShared());
    mls.setPolynomialOrder(2);
    mls.setSearchMethod(std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>(pointTree));
    mls.setSearchRadius(_searchRadius);

    // Reconstruct
    mls.process(mls_points);

    pcl::copyPointCloud(mls_points, _pointCloud);
}

void printPoint(const pcl::PointXYZRGB& _point){
    std::cout << "Point: ("
    << _point.x << ", "
    << _point.y << ", "
    << _point.z << ")" << std::endl;
}

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>& _coloredCloud,
    const pcl::RGB& _color
){
    for (auto& point : _coloredCloud.points)
    {
        point.r = _color.r;
        point.g = _color.g;
        point.b = _color.b;
    }
}

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>& _coloredCloud,
    const pcl::PointIndices& inliers,
    const pcl::RGB& _color
){
    // Assign the same color to all points in the segmented region
    for (std::size_t i = 0; i < inliers.indices.size(); ++i)
    {
        const auto& idx = inliers.indices[i];

        if(_coloredCloud.points[idx].r == _coloredCloud.points[idx].g && _coloredCloud.points[idx].g == _coloredCloud.points[idx].b)
        {
            _coloredCloud.points[idx].r = _color.r;
            _coloredCloud.points[idx].g = _color.g;
            _coloredCloud.points[idx].b = _color.b;
        }
        else
        {
            if(_coloredCloud.points[idx].r == 0){_coloredCloud.points[idx].r = _color.r;}
            if(_coloredCloud.points[idx].g == 0){_coloredCloud.points[idx].g = _color.g;}
            if(_coloredCloud.points[idx].b == 0){_coloredCloud.points[idx].b = _color.b;}
        }
    }
}

void addCloud2View(pcl::visualization::PCLVisualizer::Ptr _viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const std::string _name){
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud);
    _viewer->addPointCloud<pcl::PointXYZRGB> (_cloud, rgb, _name);
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _name);
}

void view(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> _clouds){
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    int i = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud : _clouds) {
        addCloud2View(viewer, cloud, "cloud" + std::to_string(i));
        ++i;
    }
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

    pcl::PointXYZRGB centerPoint(15.50081099, 3.76794873, 0.0, 255, 255, 255);
    pcl::search::KdTree<pcl::PointXYZRGB> kdTree;

    pcl::PointCloud<pcl::PointNormal> normalsCloud = extractNormalsPC(*cloud, pcl::PointXYZRGB(0.0, 0.0, 0.0, 255, 255, 255));
    pcl::PointXYZRGB highestPoint = getHighestPoint(*cloud);
    centerPoint.z = highestPoint.z;
    printPoint(centerPoint);

    downSamplePointCloud(*cloud, 0.1);

    kdTree.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr landingZoneCloud = extractNeighborPC(*cloud, kdTree, centerPoint, 1.5);
    colorSegmentedPoints(*landingZoneCloud, pcl::RGB(255,0,0));

    pcl::PointIndices biggestIdx = extractBiggestCluster(*cloud, 0.5, 10);
    colorSegmentedPoints(*cloud, pcl::RGB(255,255,255));
    // colorSegmentedPoints(*cloud, biggestIdx, pcl::RGB(0,0,255));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, *smoothCloud);
    downSamplePointCloud(*smoothCloud, 0.5);
    smoothPC(*smoothCloud, 3.0);
    colorSegmentedPoints(*smoothCloud, pcl::RGB(0,0,255));

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds = {landingZoneCloud, smoothCloud};
    view(clouds);
    return 0;
}
