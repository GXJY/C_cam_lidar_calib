#include <iostream>
#include <filesystem>
#include <deque>
#include <algorithm>
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"



namespace huawei{
  struct EIGEN_ALIGN16 Point{
    PCL_ADD_POINT4D;
    uint32_t time;
    float distance;
    float pitch;
    float yaw;
    uint16_t intensity;
    uint16_t ring;
    uint16_t block;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace huawei
POINT_CLOUD_REGISTER_POINT_STRUCT(huawei::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(uint32_t, time, time)(float, pitch, pitch)(float, yaw, yaw)(uint16_t, intensity, intensity)(uint16_t, ring, ring)(uint16_t, block, block))
namespace rs128{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    double time;  // float
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(rs128::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, time, timestamp))
                                  
namespace rslidar_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    double time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace rslidar_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(rslidar_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, time, time)(std::uint16_t, ring, ring))

class Data_Loader
{
public:
    Data_Loader();
    ~Data_Loader();

    uint64_t getTimeFromString(std::string ss);
    typedef pcl::PointCloud<huawei::Point> PointCloudHuawei; 
    //typedef pcl::PointCloud<velodyne_ros::Point> PointCloudHozon;
    typedef pcl::PointCloud<rs128::Point> PointCloudRS128;

    // void ReadOneLidar(std::deque<PointCloudXYZI::Ptr> &lidar_deque);
    PointCloudHuawei readOneCloud();
    PointCloudHuawei readHuaweiCloud();
    PointCloudRS128 readRS128Cloud();

    bool haveReadAllFile();
    bool haveReadAllImage;
    bool haveReadAllLidar;

    typedef std::deque<sensor_msgs::Image> ImageDequeType;
    typedef std::shared_ptr<ImageDequeType> ImageDequePtr;
    void setImageBufferPtr(ImageDequePtr &imageDqPtr);

    typedef std::deque<pcl::PCLPointCloud2> CloudDequeType;
    typedef std::shared_ptr<CloudDequeType> CloudDequePtr;
    void setLidarBufferPtr(CloudDequePtr &cloudDqPtr);

    typedef std::deque<float> TimeDequeType;
    typedef std::shared_ptr<TimeDequeType> TimeDequePtr;
    /**
    @param timeDqPtr 时间队列智能指针
    **/
    void setTimeBufferPtr(TimeDequePtr &timeDqPtr);
    bool string_contains_pcd(const std::string &str);
    bool string_contains_image(const std::string &str);
    bool file_exists(const std::string& filename);

    void getLidarFileName();
    void getImageFileName();
    void setFilePath(std::string image_folder_path,std::string pcd_folder_path);

private:
    /* data */
    
    std::string image_folder_path, pcd_folder_path;
    std::deque<std::string> image_filenames, pcd_filenames;
    ImageDequeType image_ptr;
    CloudDequePtr clouds_ptr;
    TimeDequePtr times_ptr;
    std::deque<std::string> lidarFileDeque; 
    std::deque<std::string> imageFileDeque;


};

Data_Loader::Data_Loader()
{
}

Data_Loader::~Data_Loader()
{
}


