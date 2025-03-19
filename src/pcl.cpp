#include <iostream>
#include <fstream>

#include <pcl/io/ply_io.h> // For loading PLY files
#include <pcl/point_types.h> // For point cloud types

#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/features/principal_curvatures.h>

#include <pcl/visualization/pcl_visualizer.h> // For visualization (optional)


const int N_NEIGHBORS_SEARCH = 4;
const float DRONE_RADIUS = 1.5;

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

pcl::PointIndices removeNaNFromNormalCloud(pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud){
    pcl::PointCloud<pcl::PointNormal> tempCloud;
    tempCloud.points.reserve(_normalsCloud->points.size());

    // Create a PointIndices object to store the removed indices
    pcl::PointIndices removedIndices;
    removedIndices.indices.reserve(_normalsCloud->points.size());

    int i = 0;
    for (const auto& normal : _normalsCloud->points) {
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

    pcl::copyPointCloud(tempCloud, *_normalsCloud);
    return removedIndices;
}

pcl::PointIndices computeNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const pcl::PointXYZRGB& _viewPoint,
    const int _searchNeighbors
){
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(_pointCloud);

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
    ne.setInputCloud(_pointCloud);
    ne.setViewPoint(_viewPoint.x, _viewPoint.y, _viewPoint.z);
    ne.setSearchMethod(kdTree);
    ne.setKSearch(_searchNeighbors);
    ne.compute(*_normalsCloud);

    pcl::PointIndices removedIdx = removeNaNFromNormalCloud(_normalsCloud);
    extractPoints(*_pointCloud, *_pointCloud, removedIdx, true);

    return removedIdx;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const pcl::PointXYZRGB& _centroid
){
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsPC(_pointCloud, normalsCloud, _centroid, N_NEIGHBORS_SEARCH);
    return normalsCloud;
}

pcl::PointXYZRGB getHighestPoint(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud){
    // Initialize the highest point with the first point in the cloud
    pcl::PointXYZRGB highestPoint = _pointCloud->points[0];

    // Iterate through the cloud to find the highest point
    for (const auto& point : _pointCloud->points) {
        if (point.z > highestPoint.z) {
            highestPoint = point;
        }
    }

    return highestPoint;
}

void downSamplePointCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const float _leafSize
){
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(_pointCloud);
    sor.setLeafSize(_leafSize, _leafSize, _leafSize);
    sor.filter(*_pointCloud);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractNeighborPC(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const pcl::PointXYZRGB _center,
    const float _radius
){
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_pointCloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr neighborsPC(new pcl::PointCloud<pcl::PointXYZRGB>);
    neighborsPC->points.reserve(_pointCloud->points.size());

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    if(kdtree->radiusSearch(_center, _radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i){
            neighborsPC->points.emplace_back(_pointCloud->points[pointIdxRadiusSearch[i]]);
        }
    }

    return neighborsPC;
}

pcl::PointIndices extractBiggestCluster(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const float _threshold,
    const int _minPoints
){
    // Cluster extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(_threshold); // Adjust tolerance (distance threshold)
    ec.setMinClusterSize(_minPoints); // Adjust minimum points per cluster
    ec.setMaxClusterSize(std::numeric_limits<int>::max()); // No upper limit
    ec.setInputCloud(_pointCloud);
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
        extractPoints(*_pointCloud, *_pointCloud, inliers, false);
        std::cout << "The point cloud has " << std::to_string(cluster_indices.size()) << " clusters." << std::endl;
    }
    else
    {
        std::cout << "No clusters found in the point cloud." << std::endl;
    }

    return inliers;
}

void smoothPC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud, const float _searchRadius)
{
    // Output has the PointNormal type in order to store the normals calculated by MLS
    pcl::PointCloud<pcl::PointNormal> mls_points;

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;
    
    mls.setComputeNormals(true);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_pointCloud);
    // Set parameters
    mls.setInputCloud(_pointCloud);
    mls.setPolynomialOrder(2);
    mls.setSearchMethod(kdtree);
    mls.setSearchRadius(_searchRadius);

    // Reconstruct
    mls.process(mls_points);

    // Manual copy because of different types PointNormal vs PointXYZRGB
    for (size_t i = 0; i < mls_points.size(); ++i) {
        _pointCloud->points[i].x = mls_points.points[i].x;
        _pointCloud->points[i].y = mls_points.points[i].y;
        _pointCloud->points[i].z = mls_points.points[i].z;
    }

    // pcl::copyPointCloud(mls_points, *_pointCloud);
}

float computeDensity(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, float _radius) {
    // Compute the volume of the sphere
    float volume = (4.0f / 3.0f) * M_PI * std::pow(_radius, 3);

    // Compute the density
    float density = _cloud->points.size() / volume;

    std::cout << "Density: " << density << std::endl;
    return density;
}

int projectPoint(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    pcl::PointXYZRGB& _point
){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    for (auto& point : cloud_copy->points) {
        point.z = 0;
    }

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(cloud_copy);

    std::vector<int> point_indices;
    std::vector<float> point_distances;
    int nClosestPoints = 4;
    float avgZ = 0.0;
    if (kdtree->nearestKSearch(_point, nClosestPoints, point_indices, point_distances) > 0) {
        for (auto& idx : point_indices) {
            pcl::PointXYZRGB closest_point = _cloud->points[idx];
            std::cout << "Closest point: (" << closest_point.x << ", "
                      << closest_point.y << ", " << closest_point.z << ")" << std::endl;
            std::cout << "Z coordinate: " << closest_point.z << std::endl;
            avgZ += closest_point.z;
        }
        avgZ = avgZ/float(point_indices.size());
        std::cout << "Z average coordinate: " << avgZ << std::endl;
    }

    _point.z = avgZ;

    return avgZ;
}

pcl::PrincipalCurvatures computeCurvature(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointXYZRGB& _point,
    const float _radius
){
    // Step 1: Create a copy of the input cloud and add the point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    cloud_copy->push_back(_point);

    // Step 2: Create a KdTree and set the input cloud
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(cloud_copy);

    // Step 3: Compute normals for the whole cloud
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(cloud_copy);
    ne.setSearchMethod(kdtree);
    ne.setRadiusSearch(_radius); // Use the same radius for normal estimation
    ne.compute(*normals);

    // Step 4: Compute principal curvatures for the whole cloud
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> ce;
    ce.setInputCloud(cloud_copy);
    ce.setInputNormals(normals);
    ce.setSearchMethod(kdtree);
    ce.setRadiusSearch(_radius);

    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    ce.compute(*curvatures);

    // Step 5: Print out results
    int lastIdx = curvatures->size() - 1;
    const auto& curvature = curvatures->points[lastIdx];
    float pc1 = curvature.pc1;
    float pc2 = curvature.pc2;
    float mean_curvature = (pc1 + pc2) / 2.0f;
    float gaussian_curvature = pc1 * pc2;

    std::cout << "Point " << lastIdx << ": Principal Curvatures: " << pc1 << ", " << pc2
                << ", Mean Curvature: " << mean_curvature
                << ", Gaussian Curvature: " << gaussian_curvature << std::endl;
    std::cout << "  Principal Directions: (" << curvature.principal_curvature_x << ", "
                << curvature.principal_curvature_y << ", " << curvature.principal_curvature_z << ")\n";

    // Step 6: Return the curvature of the target point (last point in the cloud)
    if (!curvatures->empty()) {
        return curvatures->points[lastIdx]; // The last point corresponds to the added point
    } else {
        throw std::runtime_error("No curvature computed for the specified point.");
    }
}

Eigen::Vector4f computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud){
    // Compute the centroid of the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*_cloud, centroid);

    // Perform PCA
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(_cloud);
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
    Eigen::Vector3f eigenvalues = pca.getEigenValues();

    // The eigenvector corresponding to the smallest eigenvalue is the normal of the plane
    Eigen::Vector3f normal = eigenvectors.col(2); // Third column (smallest eigenvalue)

    // Compute the plane coefficients (ax + by + cz + d = 0)
    float a = normal[0];
    float b = normal[1];
    float c = normal[2];
    float d = -(normal.dot(centroid.head<3>()));

    // Print the plane coefficients
    std::cout << "Plane coefficients (ax + by + cz + d = 0):\n";
    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n";
    std::cout << "c: " << c << "\n";
    std::cout << "d: " << d << "\n";

    Eigen::Vector4f coefficients;
    coefficients[0] = a;
    coefficients[1] = b;
    coefficients[2] = c;
    coefficients[3] = d;

    return coefficients;
}

float computePlaneAngle(const Eigen::Vector4f& _coefficients){
    // Extract the normal vector (nx, ny, nz)
    Eigen::Vector3f normal(_coefficients[0], _coefficients[1], _coefficients[2]);

    // Normalize the normal vector (in case it's not already normalized)
    normal.normalize();

    // Define the Z-axis vector
    Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);

    // Compute the dot product between the normal and the Z-axis
    float dot_product = normal.dot(z_axis);

    // Compute the angle in radians
    double theta = std::acos(dot_product);

    // Convert to degrees
    double theta_deg = theta * 180.0 / M_PI;

    double slope = theta_deg;
    if (theta_deg > 90.0) {
        slope = 180.0 - theta_deg;
    }

    // Print the results
    std::cout << "Angle of slope: " << slope << " degrees\n";

    return slope;
}

float pointToPlaneDistance(const pcl::PointXYZRGB& _point, const Eigen::Vector4f& _coefficients) {
    float a = _coefficients[0];
    float b = _coefficients[1];
    float c = _coefficients[2];
    float d = _coefficients[3];

    // Compute the distance
    float distance = std::abs(a * _point.x + b * _point.y + c * _point.z + d);
    return distance;
}

float computeStandardDeviation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const Eigen::Vector4f& coefficients) {
    std::vector<float> distances;
    for (const auto& point : _cloud->points) {
        float distance = pointToPlaneDistance(point, coefficients);
        distances.push_back(distance);
    }

    // Compute the mean distance
    float mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();

    // Compute the variance
    float variance = 0.0f;
    for (float distance : distances) {
        variance += std::pow(distance - mean, 2);
    }
    variance /= distances.size();

    // Compute the standard deviation
    float std_dev = std::sqrt(variance);
    std::cout << "Standard Deviation: " << std_dev << std::endl;
    return std_dev;
}

float computePointsDist2D(
    const pcl::PointXYZRGB& _point1,
    const pcl::PointXYZRGB& _point2
){
    float dx = _point1.x - _point2.x;
    float dy = _point1.y - _point2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float computePointsDist3D(
    const pcl::PointXYZRGB& _point1,
    const pcl::PointXYZRGB& _point2
){
    float dx = _point1.x - _point2.x;
    float dy = _point1.y - _point2.y;
    float dz = _point1.z - _point2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<float> computeDistToCenters(
    const pcl::PointXYZRGB& _landingPoint,
    const pcl::PointXYZRGB& _dfCenterPoint,
    const pcl::PointXYZRGB& _pcCenterPoint
) {
    std::vector<float> output(4);
    output[0] = computePointsDist2D(_landingPoint, _dfCenterPoint);
    output[1] = computePointsDist2D(_landingPoint, _pcCenterPoint);
    output[2] = computePointsDist3D(_landingPoint, _dfCenterPoint);
    output[3] = computePointsDist3D(_landingPoint, _pcCenterPoint);

    std::cout << "Distance to DF Center 2D: " << output[0] << std::endl;
    std::cout << "Distance to PC Center 2D: " << output[1] << std::endl;
    std::cout << "Distance to DF Center 3D: " << output[2] << std::endl;
    std::cout << "Distance to PC Center 3D: " << output[3] << std::endl;

    return output;
}

void saveToCSV(
    const std::string& _filename,
    const pcl::PrincipalCurvatures& _curvatures,
    const float _density,
    const float _slope,
    const float _stdDev,
    const std::vector<float>& _centerDists
) {
    // Open the file for writing
    std::ofstream file;
    file.open(_filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << _filename << " for writing." << std::endl;
        return;
    }

    // Write headers
    file << "Curvature_PC1,Curvature_PC2,Mean_Curvature,Gaussian_Curvature,"
         << "Density,Slope,Standard_Deviation,"
         << "Distance_DF_Center_2D,Distance_PC_Center_2D,Distance_DF_Center_3D,Distance_PC_Center_3D\n";

    // Write data
    file << _curvatures.pc1 << "," << _curvatures.pc2 << "," << (_curvatures.pc1 + _curvatures.pc2) / 2.0f << "," << _curvatures.pc1 * _curvatures.pc2 << ","
         << _density << ","
         << _slope << ","
         << _stdDev << ","
         << _centerDists[0] << "," << _centerDists[1] << "," << _centerDists[2] << "," << _centerDists[3] << "\n";

    // Close the file
    file.close();

    std::cout << "Data saved to " << _filename << std::endl;
}

void printPoint(const pcl::PointXYZRGB& _point){
    std::cout << "Point: ("
    << _point.x << ", "
    << _point.y << ", "
    << _point.z << ")" << std::endl;
}

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _coloredCloud,
    const pcl::RGB& _color
){
    for (auto& point : _coloredCloud->points)
    {
        point.r = _color.r;
        point.g = _color.g;
        point.b = _color.b;
    }
}

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _coloredCloud,
    const pcl::PointIndices& inliers,
    const pcl::RGB& _color
){
    // Assign the same color to all points in the segmented region
    for (std::size_t i = 0; i < inliers.indices.size(); ++i)
    {
        const auto& idx = inliers.indices[i];

        if(_coloredCloud->points[idx].r == _coloredCloud->points[idx].g && _coloredCloud->points[idx].g == _coloredCloud->points[idx].b)
        {
            _coloredCloud->points[idx].r = _color.r;
            _coloredCloud->points[idx].g = _color.g;
            _coloredCloud->points[idx].b = _color.b;
        }
        else
        {
            if(_coloredCloud->points[idx].r == 0){_coloredCloud->points[idx].r = _color.r;}
            if(_coloredCloud->points[idx].g == 0){_coloredCloud->points[idx].g = _color.g;}
            if(_coloredCloud->points[idx].b == 0){_coloredCloud->points[idx].b = _color.b;}
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
    // spin() instead of spinOnce() avoids crash
    viewer->spin();
}

int main() {
    std::string ply_file_path = "../inputs/rtabmap_cloud.ply";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ogCloud = loadPly(ply_file_path);
    // pcl::PointCloud<pcl::PointNormal> normalsCloud = extractNormalsPC(*cloud, pcl::PointXYZRGB(0.0, 0.0, 0.0, 255, 255, 255));

    // Landing point [15.50081099  3.76794873 12.62245941]

    pcl::PointXYZRGB dfCenterPoint(15.50081099, 3.76794873, 0.0, 255, 255, 255);
    pcl::PointXYZRGB landingPoint(15.0, 4.0, 0.0, 255, 255, 255);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*ogCloud));
    downSamplePointCloud(cloud, 0.1);
    extractBiggestCluster(cloud, 0.5, 10);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud));
    downSamplePointCloud(smoothCloud, 0.5);
    smoothPC(smoothCloud, 2*DRONE_RADIUS);
    projectPoint(smoothCloud, dfCenterPoint);
    projectPoint(smoothCloud, landingPoint);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr landingCloud = extractNeighborPC(ogCloud, landingPoint, DRONE_RADIUS);

    pcl::PointXYZRGB pcCenterPoint = getHighestPoint(smoothCloud);

    pcl::PrincipalCurvatures curvatures = computeCurvature(smoothCloud, landingPoint, DRONE_RADIUS);
    float density = computeDensity(landingCloud, DRONE_RADIUS);
    Eigen::Vector4f coef = computePlane(landingCloud);
    float slope = computePlaneAngle(coef);
    float stdDev = computeStandardDeviation(landingCloud, coef);
    std::vector<float> centerDists = computeDistToCenters(landingPoint, dfCenterPoint, pcCenterPoint);

    saveToCSV("../outputs/test.csv", curvatures, density, slope, stdDev, centerDists);

    colorSegmentedPoints(cloud, pcl::RGB(255,255,255));
    colorSegmentedPoints(smoothCloud, pcl::RGB(0,0,255));
    colorSegmentedPoints(landingCloud, pcl::RGB(255,0,0));
    view(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>{ogCloud, smoothCloud, landingCloud});
    return 0;
}
