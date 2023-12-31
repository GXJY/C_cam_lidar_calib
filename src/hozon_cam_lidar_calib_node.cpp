//
// Created by usl on 4/6/19.
//

#include <algorithm>
#include <random>
#include <chrono>
#include <ctime>



#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>

#include <calibration_error_term.h>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "ceres/rotation.h"
#include "ceres/covariance.h"
#include "data_loader.h"

#include <fstream>
#include <iostream>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                        sensor_msgs::Image> SyncPolicy;

std::deque<double> time_buffer;

std::deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> lidar_buffer;
std::deque<sensor_msgs::Image::ConstPtr> image_buffer;

class camLidarCalib {
private:
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;
    ros::Publisher cloud_pub;

    cv::Mat image_in;
    cv::Mat image_read;
    cv::Mat image_resized;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> projected_points;
    bool boardDetectedInCam;
    double dx, dy;
    int checkerboard_rows, checkerboard_cols;
    int min_points_on_plane;
    cv::Mat tvec, rvec;
    cv::Mat C_R_W;
    Eigen::Matrix3d c_R_w;
    Eigen::Vector3d c_t_w;
    Eigen::Vector3d r3;
    Eigen::Vector3d r3_old;
    Eigen::Vector3d Nc;

    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<std::vector<Eigen::Vector3d> > all_lidar_points;
    std::vector<Eigen::Vector3d> all_normals;

    std::string result_str, result_rpy, result_sigma;

    std::string camera_in_topic;
    std::string lidar_in_topic;

    int num_views;

    std::string cam_config_file_path;
    std::string image_folder_path, pcd_folder_path;

    int image_width, image_height;
    double x_min, x_max;
    double y_min, y_max;
    double z_min, z_max;
    double ransac_threshold;
    int no_of_initializations;
    std::string initializations_file;
    std::ofstream init_file;


public:
 
    camLidarCalib(ros::NodeHandle n) {
        nh = n;     

        // {
        //     ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("image_topic", 10);
        //     ros::Publisher pcd_pub = nh.advertise<sensor_msgs::Image>("pcd_topic", 10);

        //     cv::Mat image;
        //     cv_bridge::CvImage cv_image;
        //     for(int i = 0; i < max_file_num; ++i) {
        //         if (!image_files.empty())
        //         {
        //             image = cv::imread(image_files.front());
        //             // 将OpenCV图像转换为ROS图像消息
        //             cv_image.encoding = "bgr8";
        //             cv_image.image = image;
        //             sensor_msgs::ImagePtr msg = cv_image.toImageMsg();

        //             // 发布图像消息
        //             image_pub.publish(msg);

        //             ROS_INFO_STREAM("Published image: " << image_files.front());
        //             image_files.pop_front();
        //             // ros::Duration(1.0).sleep();  // 等待1秒
        //         }
        //         if(!pcd_files.empty()) {
        //             pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //             pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_files.front(), *cloud); 

        //             sensor_msgs::PointCloud2 output;
        //             pcl::toROSMsg(*cloud, output);
        //             output.header.frame_id = "base_link"; // 设置点云的坐标系

        //             pcd_pub.publish(output);
        //             ROS_INFO_STREAM("Published pcd: " << pcd_files.front());
        //             pcd_files.pop_front();
        //             // ros::Duration(1.0).sleep();  // 等待1秒  
        //         }
        //     }

        //     // 获取文件夹中的所有图像文件
        //     boost::filesystem::directory_iterator image_end_itr;
        //     for (boost::filesystem::directory_iterator i_itr(image_folder_path); i_itr != image_end_itr; ++i_itr)
        //     {
        //        if (boost::filesystem::is_regular_file(i_itr->path()))
        //        {
        //             // 读取图像文件
        //             image = cv::imread(i_itr->path().string());
        //             if (!image.empty())
        //             {
        //                 // 将OpenCV图像转换为ROS图像消息
        //                 cv_image.encoding = "bgr8";
        //                 cv_image.image = image;
        //                 sensor_msgs::ImagePtr msg = cv_image.toImageMsg();

        //                 // 发布图像消息
        //                 image_pub.publish(msg);

        //                 ROS_INFO_STREAM("Published image: " << i_itr->path().string());
        //                 // ros::Duration(1.0).sleep();  // 等待1秒
        //             }
        //         }
        //     }
        //     boost::filesystem::directory_iterator pcd_end_itr;
        //     for (boost::filesystem::directory_iterator l_itr(pcd_folder_path); l_itr != pcd_end_itr; ++l_itr)
        //     {
        //        if (boost::filesystem::is_regular_file(l_itr->path()))
        //        {
        //             pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //             pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_folder_path + l_itr->path().string(), *cloud); 

        //             sensor_msgs::PointCloud2 output;
        //             pcl::toROSMsg(*cloud, output);
        //             output.header.frame_id = "base_link"; // 设置点云的坐标系

        //             pcd_pub.publish(output);
        //             ROS_INFO_STREAM("Published pcd: " << l_itr->path().string());
        //             // ros::Duration(1.0).sleep();  // 等待1秒    
        //         }
        //     }
        // }
    

        // cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "pcd_topic", 1);
        // image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, "image_topic", 1);
        // sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *cloud_sub, *image_sub);
        // sync->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2));
        // cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("points_out", 1);
        
        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(5, 1, CV_64F);
        boardDetectedInCam = false;
        tvec = cv::Mat::zeros(3, 1, CV_64F);
        rvec = cv::Mat::zeros(3, 1, CV_64F);
        C_R_W = cv::Mat::eye(3, 3, CV_64F);
        c_R_w = Eigen::Matrix3d::Identity();

        dx = readParam<double>(nh, "dx");
        dy = readParam<double>(nh, "dy");
        checkerboard_rows = readParam<int>(nh, "checkerboard_rows");
        checkerboard_cols = readParam<int>(nh, "checkerboard_cols");
        min_points_on_plane = readParam<int>(nh, "min_points_on_plane");
        num_views = readParam<int>(nh, "num_views");
        no_of_initializations = readParam<int>(nh, "no_of_initializations");
        initializations_file = readParam<std::string>(nh, "initializations_file");
        for(int i = 0; i < checkerboard_rows; i++)
            for (int j = 0; j < checkerboard_cols; j++)
                object_points.emplace_back(cv::Point3f(i*dx, j*dy, 0.0));


        cam_config_file_path = readParam<std::string>(nh, "cam_config_file_path");
        readCameraParams(cam_config_file_path,
                        image_height,
                        image_width,
                        distCoeff,
                        projection_matrix);
        x_min = readParam<double>(nh, "x_min");
        x_max = readParam<double>(nh, "x_max");
        y_min = readParam<double>(nh, "y_min");
        y_max = readParam<double>(nh, "y_max");
        z_min = readParam<double>(nh, "z_min");
        z_max = readParam<double>(nh, "z_max");
        ransac_threshold = readParam<double>(nh, "ransac_threshold");

        {
            result_str = readParam<std::string>(nh, "result_file");
            result_rpy = readParam<std::string>(nh, "result_rpy_file");
            result_sigma = readParam<std::string>(nh, "result_sigma_file");
            // 每次开始前清除文件内容
            std::ofstream str_clear(result_str,std::ios::out|std::ios::trunc);  // 创建新文件
            if (str_clear.is_open()) {          // 检查是否成功打开
                str_clear<< std::endl;
                str_clear.close();               // 关闭文件流
                std::cout << "str File created successfully." << std::endl;
            } else {
                std::cout << "Failed to create str file." << std::endl;
            }

            std::ofstream rpy_clear(result_rpy,std::ios::out|std::ios::trunc);  // 创建新文件
            if (rpy_clear.is_open()) {          // 检查是否成功打开
                rpy_clear<< std::endl;
                rpy_clear.close();               // 关闭文件流
                std::cout << "rpy File created successfully." << std::endl;
            } else {
                std::cout << "Failed to create rpy file." << std::endl;
            }

            std::ofstream sigma_clear(result_sigma,std::ios::out|std::ios::trunc);  // 创建新文件
            if (sigma_clear.is_open()) {          // 检查是否成功打开
                sigma_clear<< std::endl;
                sigma_clear.close();               // 关闭文件流
                std::cout << "sigma File created successfully." << std::endl;
            } else {
                std::cout << "Failed to create sigma file." << std::endl;
            }
        }


        image_folder_path = readParam<std::string>(nh, "image_folder_path");
        pcd_folder_path = readParam<std::string>(nh, "pcd_folder_path");

        std::deque<std::string> image_files, pcd_files;

        boost::filesystem::directory_iterator image_end_itr, pcd_end_itr;
        
        for (boost::filesystem::directory_iterator i_itr(image_folder_path); i_itr != image_end_itr; ++i_itr)
        {
           if (boost::filesystem::is_regular_file(i_itr->path()))
           {
                image_files.push_back(i_itr->path().string());
           }
        }
        for (boost::filesystem::directory_iterator j_itr(pcd_folder_path); j_itr != pcd_end_itr; ++j_itr)
        {
           if (boost::filesystem::is_regular_file(j_itr->path()))
           {
                pcd_files.push_back(j_itr->path().string());
           }
        }
        sort(image_files.begin(), image_files.end());
        sort(pcd_files.begin(), pcd_files.end());
        int max_file_num = std::max(image_files.size(), pcd_files.size());


        while(!image_files.empty() && !pcd_files.empty())
        {

            ROS_WARN_STREAM(image_files.front() << " " << pcd_files.front());
            std::string im = image_h(image_files.front());
            std::string pcd =  pcd_h(pcd_files.front());
            if(runSolver(im, pcd)) {
                std::cout << "end end end" << std::endl;
                break;
            }
            // image_files.pop_front();
            // image_files.pop_front();
            image_files.pop_front();
            pcd_files.pop_front();
        }
    }

    std::string image_h(std::string &image_name)
    {
        std::cout << "start image_h" << std::endl;
        image_in = cv::imread(image_name, cv::IMREAD_COLOR);

        std::cout << "image in" << std::endl;
        // image_in = cv_bridge::toCvShare(image_read, "bgr8")->image;
        boardDetectedInCam = cv::findChessboardCorners(image_in,
                                                        cv::Size(checkerboard_cols, checkerboard_rows),
                                                        image_points,
                                                        cv::CALIB_CB_ADAPTIVE_THRESH+
                                                        cv::CALIB_CB_NORMALIZE_IMAGE);
        if (boardDetectedInCam) {
            std::cout << "board detect" << std::endl;
        } 
        else {
            std::cout << "board not detect   " << image_name << std::endl;
        }
        cv::drawChessboardCorners(image_in,
                                    cv::Size(checkerboard_cols, checkerboard_rows),
                                    image_points,
                                    boardDetectedInCam);
        if(image_points.size() == object_points.size()){

            // cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false, CV_ITERATIVE);
            cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false);
            
            projected_points.clear();
            cv::projectPoints(object_points, rvec, tvec, projection_matrix, distCoeff, projected_points, cv::noArray());
            for(int i = 0; i < projected_points.size(); i++){
                cv::circle(image_in, projected_points[i], 16, cv::Scalar(0, 255, 0), 10, cv::LINE_AA, 0);
            }
            cv::Rodrigues(rvec, C_R_W);
            cv::cv2eigen(C_R_W, c_R_w);
            c_t_w = Eigen::Vector3d(tvec.at<double>(0),
                                    tvec.at<double>(1),
                                    tvec.at<double>(2));

            r3 = c_R_w.block<3,1>(0,2);
            Nc = (r3.dot(c_t_w))*r3;
            std::cout << "pnp sloved" << std::endl;
        }
        cv::resize(image_in, image_resized, cv::Size(), 0.25, 0.25);
        std::cout << "view board" << std::endl;
        cv::imshow("view", image_resized);
        cv::waitKey(10);
        std::cout << "end image_h" << std::endl;
        return image_name;

    }

    std::string pcd_h(std::string &pcd_name)
    {
        std::cout << "start pcd_h" << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_name, *in_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        /// Pass through filters
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(in_cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min, x_max);
        pass_x.filter(*cloud_filtered_x);

        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud_filtered_x);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min, y_max);
        pass_y.filter(*cloud_filtered_y);

        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(cloud_filtered_y);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min, z_max);
        pass_z.filter(*cloud_filtered_z);

        /// Plane Segmentation
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_z));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
        ransac.setDistanceThreshold(ransac_threshold);
        ransac.computeModel();
        std::vector<int> inliers_indicies;
        ransac.getInliers(inliers_indicies);
        pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_z, inliers_indicies, *plane);

        /// Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(plane);
        sor.setMeanK (50);
        sor.setStddevMulThresh (1);
        sor.filter (*plane_filtered);

        /// Store the points lying in the filtered plane in a vector
        lidar_points.clear();
        for (size_t i = 0; i < plane_filtered->points.size(); i++) {
            double X = plane_filtered->points[i].x;
            double Y = plane_filtered->points[i].y;
            double Z = plane_filtered->points[i].z;
            lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
        }
        std::cout << "end pcd_h" << std::endl;
        return pcd_name;
    }
    

    void readCameraParams(std::string cam_config_file_path,
                          int &image_height,
                          int &image_width,
                          cv::Mat &D,
                          cv::Mat &K) {
        cv::FileStorage fs_cam_config(cam_config_file_path, cv::FileStorage::READ);
        if(!fs_cam_config.isOpened())
            std::cerr << "Error: Wrong path: " << cam_config_file_path << std::endl;
        fs_cam_config["image_height"] >> image_height;
        fs_cam_config["image_width"] >> image_width;
        fs_cam_config["k1"] >> D.at<double>(0);
        fs_cam_config["k2"] >> D.at<double>(1);
        fs_cam_config["p1"] >> D.at<double>(2);
        fs_cam_config["p2"] >> D.at<double>(3);
        fs_cam_config["k3"] >> D.at<double>(4);
        fs_cam_config["fx"] >> K.at<double>(0, 0);
        fs_cam_config["fy"] >> K.at<double>(1, 1);
        fs_cam_config["cx"] >> K.at<double>(0, 2);
        fs_cam_config["cy"] >> K.at<double>(1, 2);
    }

    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name)
    {
        T ans;
        if (n.getParam(name, ans))
        {
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        }
        else
        {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    void addGaussianNoise(Eigen::Matrix4d &transformation) {
        std::vector<double> data_rot = {0, 0, 0};
        const double mean_rot = 0.0;
        std::default_random_engine generator_rot;
        generator_rot.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(mean_rot, 90);
        // std::normal_distribution<double> dist(mean_rot, 5);

        // Add Gaussian noise
        for (auto& x : data_rot) {
            x = x + dist(generator_rot);
        }

        double roll = data_rot[0]*M_PI/180;
        double pitch = data_rot[1]*M_PI/180;
        double yaw = data_rot[2]*M_PI/180;

        Eigen::Matrix3d m;
        m = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(pitch,  Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

        std::vector<double> data_trans = {0, 0, 0};
        const double mean_trans = 0.0;
        std::default_random_engine generator_trans;
        generator_trans.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist_trans(mean_trans, 0.5);

        // Add Gaussian noise
        for (auto& x : data_trans) {
            x = x + dist_trans(generator_trans);
        }

        Eigen::Vector3d trans;
        trans(0) = data_trans[0];
        trans(1) = data_trans[1];
        trans(2) = data_trans[2];

        Eigen::Matrix4d trans_noise = Eigen::Matrix4d::Identity();
        trans_noise.block(0, 0, 3, 3) = m;
        trans_noise.block(0, 3, 3, 1) = trans;
        transformation = transformation*trans_noise;
    }

//     void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {

//         pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::fromROSMsg(*cloud_msg, *in_cloud);

//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::PointCloud<pcl::PointXYZ >::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
//         pcl::PointCloud<pcl::PointXYZ >::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

//         /// Pass through filters
//         pcl::PassThrough<pcl::PointXYZ> pass_x;
//         pass_x.setInputCloud(in_cloud);
//         pass_x.setFilterFieldName("x");
//         pass_x.setFilterLimits(x_min, x_max);
//         pass_x.filter(*cloud_filtered_x);

//         pcl::PassThrough<pcl::PointXYZ> pass_y;
//         pass_y.setInputCloud(cloud_filtered_x);
//         pass_y.setFilterFieldName("y");
//         pass_y.setFilterLimits(y_min, y_max);
//         pass_y.filter(*cloud_filtered_y);

//         pcl::PassThrough<pcl::PointXYZ> pass_z;
//         pass_z.setInputCloud(cloud_filtered_y);
//         pass_z.setFilterFieldName("z");
//         pass_z.setFilterLimits(z_min, z_max);
//         pass_z.filter(*cloud_filtered_z);

//         /// Plane Segmentation
//         pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
//                 new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_z));
//         pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
//         ransac.setDistanceThreshold(ransac_threshold);
//         ransac.computeModel();
//         std::vector<int> inliers_indicies;
//         ransac.getInliers(inliers_indicies);
//         pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_z, inliers_indicies, *plane);

//         /// Statistical Outlier Removal
//         pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
//         sor.setInputCloud(plane);
//         sor.setMeanK (50);
//         sor.setStddevMulThresh (1);
//         sor.filter (*plane_filtered);

//         /// Store the points lying in the filtered plane in a vector
//         lidar_points.clear();
//         for (size_t i = 0; i < plane_filtered->points.size(); i++) {
//             double X = plane_filtered->points[i].x;
//             double Y = plane_filtered->points[i].y;
//             double Z = plane_filtered->points[i].z;
//             lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
//         }
// //        ROS_INFO_STREAM("No of planar_pts: " << lidar_points.size());
//         ROS_WARN_STREAM("No of planar_pts: " << plane_filtered->points.size());
// //        sensor_msgs::PointCloud2 out_cloud;
// //        pcl::toROSMsg(*plane_filtered, out_cloud);
// //        out_cloud.header.frame_id = cloud_msg->header.frame_id;
// //        out_cloud.header.stamp = cloud_msg->header.stamp;
// //        cloud_pub.publish(out_cloud);
//     }

//     void imageHandler(const sensor_msgs::ImageConstPtr &image_msg) {
//         try {
//             image_in = cv_bridge::toCvShare(image_msg, "bgr8")->image;
//             boardDetectedInCam = cv::findChessboardCorners(image_in,
//                                                            cv::Size(checkerboard_cols, checkerboard_rows),
//                                                            image_points,
//                                                            cv::CALIB_CB_ADAPTIVE_THRESH+
//                                                            cv::CALIB_CB_NORMALIZE_IMAGE);
//             cv::drawChessboardCorners(image_in,
//                                       cv::Size(checkerboard_cols, checkerboard_rows),
//                                       image_points,
//                                       boardDetectedInCam);
//             if(image_points.size() == object_points.size()){

//                 // cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false, CV_ITERATIVE);
//                 cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false);
                
//                 projected_points.clear();
//                 cv::projectPoints(object_points, rvec, tvec, projection_matrix, distCoeff, projected_points, cv::noArray());
//                 for(int i = 0; i < projected_points.size(); i++){
//                     cv::circle(image_in, projected_points[i], 16, cv::Scalar(0, 255, 0), 10, cv::LINE_AA, 0);
//                 }
//                 cv::Rodrigues(rvec, C_R_W);
//                 cv::cv2eigen(C_R_W, c_R_w);
//                 c_t_w = Eigen::Vector3d(tvec.at<double>(0),
//                                         tvec.at<double>(1),
//                                         tvec.at<double>(2));

//                 r3 = c_R_w.block<3,1>(0,2);
//                 Nc = (r3.dot(c_t_w))*r3;
//             }
//             cv::resize(image_in, image_resized, cv::Size(), 0.25, 0.25);
//             cv::imshow("view", image_resized);
//             cv::waitKey(10);
//         } catch (cv_bridge::Exception& e) {
//             ROS_ERROR("Could not convert from '%s' to 'bgr8'.",
//                       image_msg->encoding.c_str());
//         }
//     }

    bool runSolver(std::string &image_name, std::string &pcd_name) {
        std::cout << "start solve" << lidar_points.size() << std::endl;
        if (lidar_points.size() > min_points_on_plane && boardDetectedInCam) {
            std::cout << "r3.dot(r3_old)  " << r3.dot(r3_old) << std::endl;
            if (r3.dot(r3_old) < 0.90) {
                r3_old = r3;
                all_normals.push_back(Nc);
                all_lidar_points.push_back(lidar_points);
                ROS_ASSERT(all_normals.size() == all_lidar_points.size());
                std::cout << "Recording View number: " << all_normals.size() << std::endl;
                std::cout << image_name << std::endl;
                std::cout << pcd_name << std::endl;
                if (all_normals.size() >= num_views) {
                    std::cout << "Starting optimization..." << std::endl;
                    // init_file.open(initializations_file);

                    for(int counter = 0; counter < no_of_initializations; counter++) {
                        /// Start Optimization here
                        std::cout << "solve num  " << counter + 1 << " of " << no_of_initializations << std::endl;

                        /// Step 1: Initialization
                        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
                        addGaussianNoise(transformation_matrix);
                        Eigen::Matrix3d Rotn = transformation_matrix.block(0, 0, 3, 3);
                        Eigen::Vector3d axis_angle;
                        ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());

                        Eigen::Vector3d Translation =transformation_matrix.block(0, 3, 3, 1);

                        Eigen::Vector3d rpy_init = Rotn.eulerAngles(2, 1, 0)*180/M_PI;
                        Eigen::Vector3d tran_init = transformation_matrix.block(0, 3, 3, 1);

                        Eigen::VectorXd R_t(6);
                        R_t(0) = axis_angle(0);
                        R_t(1) = axis_angle(1);
                        R_t(2) = axis_angle(2);
                        R_t(3) = Translation(0);
                        R_t(4) = Translation(1);
                        R_t(5) = Translation(2);
                        std::cout << "end solve initial" << std::endl;
                        /// Step2: Defining the Loss function (Can be NULL)
//                    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
//                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                        ceres::LossFunction *loss_function = NULL;

                        /// Step 3: Form the Optimization Problem
                        ceres::Problem problem;
                        problem.AddParameterBlock(R_t.data(), 6);
                        for (int i = 0; i < all_normals.size(); i++) {
                            Eigen::Vector3d normal_i = all_normals[i];
                            std::vector<Eigen::Vector3d> lidar_points_i
                                    = all_lidar_points[i];
                            for (int j = 0; j < lidar_points_i.size(); j++) {
                                Eigen::Vector3d lidar_point = lidar_points_i[j];
                                ceres::CostFunction *cost_function = new
                                        ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>
                                        (new CalibrationErrorTerm(lidar_point, normal_i));
                                problem.AddResidualBlock(cost_function, loss_function, R_t.data());
                            }
                        }
                        std::cout << "end Form the Optimization Problem" << std::endl;


                        /// Step 4: Solve it
                        ceres::Solver::Options options;
                        options.max_num_iterations = 200;
                        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                        options.minimizer_progress_to_stdout = false;
                        ceres::Solver::Summary summary;
                        std::cout << "start solve" << std::endl;
                        ceres::Solve(options, &problem, &summary);
//                        std::cout << summary.FullReport() << '\n';
                        std::cout << "end solve" << std::endl;



                        /// Printing and Storing C_T_L in a file
                        ceres::AngleAxisToRotationMatrix(R_t.data(), Rotn.data());
                        Eigen::MatrixXd C_T_L(3, 4);
                        C_T_L.block(0, 0, 3, 3) = Rotn;
                        C_T_L.block(0, 3, 3, 1) = Eigen::Vector3d(R_t[3], R_t[4], R_t[5]);
                        std::cout << "RPY = " << Rotn.eulerAngles(2, 1, 0)*180/M_PI << std::endl;
                        std::cout << "t = " << C_T_L.block(0, 3, 3, 1) << std::endl;

                        // init_file << rpy_init(0) << "," << rpy_init(1) << "," << rpy_init(2) << ","
                        //           << tran_init(0) << "," << tran_init(1) << "," << tran_init(2) << "\n";
                        // init_file << Rotn.eulerAngles(0, 1, 2)(0)*180/M_PI << "," << Rotn.eulerAngles(0, 1, 2)(1)*180/M_PI << "," << Rotn.eulerAngles(0, 1, 2)(2)*180/M_PI << ","
                        //           << R_t[3] << "," << R_t[4] << "," << R_t[5] << "\n";

                        std::cout << rpy_init(0) << "," << rpy_init(1) << "," << rpy_init(2) << ","
                                  << tran_init(0) << "," << tran_init(1) << "," << tran_init(2) << std::endl;
                        std::cout << Rotn.eulerAngles(2, 1, 0)(0)*180/M_PI << "," << Rotn.eulerAngles(2, 1, 0)(1)*180/M_PI << "," << Rotn.eulerAngles(2, 1, 0)(2)*180/M_PI << ","
                                  << R_t[3] << "," << R_t[4] << "," << R_t[5] << std::endl;

                        /// Step 5: Covariance Estimation
                        std::cout << "start Covariance Estimation" << std::endl;
                        ceres::Covariance::Options options_cov;
                        ceres::Covariance covariance(options_cov);
                        std::vector<std::pair<const double*, const double*> > covariance_blocks;
                        covariance_blocks.push_back(std::make_pair(R_t.data(), R_t.data()));
                        CHECK(covariance.Compute(covariance_blocks, &problem));
                        double covariance_xx[6 * 6];
                        covariance.GetCovarianceBlock(R_t.data(),
                                                      R_t.data(),
                                                      covariance_xx);

                        Eigen::MatrixXd cov_mat_RotTrans(6, 6);
                        cv::Mat cov_mat_cv = cv::Mat(6, 6, CV_64F, &covariance_xx);
                        cv::cv2eigen(cov_mat_cv, cov_mat_RotTrans);

                        Eigen::MatrixXd cov_mat_TransRot(6, 6);
                        cov_mat_TransRot.block(0, 0, 3, 3) = cov_mat_RotTrans.block(3, 3, 3, 3);
                        cov_mat_TransRot.block(3, 3, 3, 3) = cov_mat_RotTrans.block(0, 0, 3, 3);
                        cov_mat_TransRot.block(0, 3, 3, 3) = cov_mat_RotTrans.block(3, 0, 3, 3);
                        cov_mat_TransRot.block(3, 0, 3, 3) = cov_mat_RotTrans.block(0, 3, 3, 3);

                        double  sigma_xx = sqrt(cov_mat_TransRot(0, 0));
                        double  sigma_yy = sqrt(cov_mat_TransRot(1, 1));
                        double  sigma_zz = sqrt(cov_mat_TransRot(2, 2));

                        double sigma_rot_xx = sqrt(cov_mat_TransRot(3, 3));
                        double sigma_rot_yy = sqrt(cov_mat_TransRot(4, 4));
                        double sigma_rot_zz = sqrt(cov_mat_TransRot(5, 5));

                        // std::vector<double> cov_sigma;
                        // cov_sigma.push_back(sigma_xx);
                        // cov_sigma.push_back(sigma_yy);
                        // cov_sigma.push_back(sigma_zz);
                        // cov_sigma.push_back(sigma_rot_xx);
                        // cov_sigma.push_back(sigma_rot_yy);
                        // cov_sigma.push_back(sigma_rot_zz);
                        std::cout << "sigma_xx = " << sigma_xx << "\t"
                                  << "sigma_yy = " << sigma_yy << "\t"
                                  << "sigma_zz = " << sigma_zz << std::endl;

                        std::cout << "sigma_rot_xx = " << sigma_rot_xx*180/M_PI << "\t"
                                  << "sigma_rot_yy = " << sigma_rot_yy*180/M_PI << "\t"
                                  << "sigma_rot_zz = " << sigma_rot_zz*180/M_PI << std::endl;
                        std::cout << "end Covariance Estimation" << std::endl;

                        std::ofstream results;
                        results.open(result_str, std::ios::out|std::ios::app);
                        results << C_T_L << "\n\n";
                        results.close();
                        
                        std::ofstream results_rpy;
                        results_rpy.open(result_rpy, std::ios::out|std::ios::app);
                        results_rpy << Rotn.eulerAngles(2, 1, 0)*180/M_PI << "\n" << C_T_L.block(0, 3, 3, 1) << "\n\n";
                        results_rpy.close();
                        
                        std::ofstream results_sigma;
                        results_sigma.open(result_sigma, std::ios::out|std::ios::app);
                        results_sigma << sigma_xx << " " << sigma_yy << " " << sigma_zz << "\n"
                                      << sigma_rot_xx << " " << sigma_rot_yy << " " << sigma_rot_zz << "\n\n" ;
                        results_sigma.close();
                        

                        std::cout << "No of initialization: " << counter << std::endl;
                    }
                    // ROS_WARN_STREAM("end solve all");
                    std::cout << "end solve all" << std::endl;
                    return true;
                    
                    // nh.shutdown();
                    // ros::shutdown();

                }
            } else {
                std::cout << "Not enough Rotation, view not recorded" << std::endl;
            }
        } else {
            if(!boardDetectedInCam)
                std::cout << "Checker-board not detected in Image." << std::endl;
            else {
                ROS_WARN_STREAM("Checker Board Detected in Image?: " << boardDetectedInCam << "\t" <<
                "No of LiDAR pts: " << lidar_points.size() << " (Check if this is less than threshold) ");
            }
        }
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg) {
        ROS_WARN_STREAM("reciving image&pcd ");
        // imageHandler(image_msg);
        // cloudHandler(cloud_msg);
        // runSolver();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "hozon_CameraLidarCalib_node");
    ros::NodeHandle nh("~");
    camLidarCalib cLC(nh);
    // ros::spin();
    return 0;
}