//
// Created by usl on 4/6/19.
//

#include <calibration_error_term.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/features/boundary.h>

#include <Eigen/Dense>
// #include <Eigen/Core>
// #include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <random>

#include "ceres/ceres.h"
#include "ceres/covariance.h"
#include "ceres/rotation.h"
#include "data_loader.h"
#include "glog/logging.h"
#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, sensor_msgs::Image>
    SyncPolicy;

std::deque<double> time_buffer;

std::deque<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> lidar_buffer;
std::deque<sensor_msgs::Image::ConstPtr> image_buffer;

class camLidarCalib {
 private:
  ros::NodeHandle nh;

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
  Eigen::Vector3d e_tvec, e_rvec;
  Eigen::Vector3d e_tvec_v, e_rvec_v;
  cv::Mat C_R_W;
  Eigen::Matrix3d c_R_w;
  Eigen::Vector3d c_t_w;
  Eigen::Vector3d r3;
  Eigen::Vector3d r3_old;
  Eigen::Vector3d Nc;
  // Eigen::MatrixXd C_T_L(3, 4);
  Eigen::Matrix4d C_T_L = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d C_T_L_tf = Eigen::Matrix4d::Identity();

  std::vector<Eigen::Vector3d> lidar_points;
  std::vector<std::vector<Eigen::Vector3d> > all_lidar_points;
  std::vector<Eigen::Vector3d> all_normals;

  std::string result_str, result_rpy, result_sigma;

  std::string camera_in_topic;
  std::string lidar_in_topic;

  int num_views;

  std::deque<std::string> image_files, pcd_files;

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

  std::vector<cv::Point3d> objectPoints_L, objectPoints_C;
  std::vector<cv::Point2d> imagePoints;
  pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl;

  Eigen::Matrix4f f_cam_vehicle_tf;
  Eigen::Matrix4d d_cam_vehicle_tf;
  Eigen::Matrix4f cam_transformation_matrix;  // = Eigen::Matrix4f::Identity();

 public:
  camLidarCalib(ros::NodeHandle n) {
    nh = n;
    f_cam_vehicle_tf << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;

    d_cam_vehicle_tf << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;

    projection_matrix = cv::Mat::eye(3, 3, CV_64F);
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
    for (int i = 0; i < checkerboard_rows; i++)
      for (int j = 0; j < checkerboard_cols; j++)
        object_points.emplace_back(cv::Point3f(i * dx, j * dy, 0.0));

    cam_config_file_path = readParam<std::string>(nh, "cam_config_file_path");
    readCameraParams(cam_config_file_path, image_height, image_width, distCoeff,
                     projection_matrix);
    x_min = readParam<double>(nh, "x_min");
    x_max = readParam<double>(nh, "x_max");
    y_min = readParam<double>(nh, "y_min");
    y_max = readParam<double>(nh, "y_max");
    z_min = readParam<double>(nh, "z_min");
    z_max = readParam<double>(nh, "z_max");
    ransac_threshold = readParam<double>(nh, "ransac_threshold");

    {  // TODO: 函数
      result_str = readParam<std::string>(nh, "result_file");
      result_rpy = readParam<std::string>(nh, "result_rpy_file");
      result_sigma = readParam<std::string>(nh, "result_sigma_file");
      // 每次开始前清除文件内容
      std::ofstream str_clear(result_str,
                              std::ios::out | std::ios::trunc);  // 创建新文件
      if (str_clear.is_open()) {  // 检查是否成功打开
        str_clear << std::endl;
        str_clear.close();  // 关闭文件流
        std::cout << "str File created successfully." << std::endl;
      } else {
        std::cout << "Failed to create str file." << std::endl;
      }

      std::ofstream rpy_clear(result_rpy,
                              std::ios::out | std::ios::trunc);  // 创建新文件
      if (rpy_clear.is_open()) {  // 检查是否成功打开
        rpy_clear << std::endl;
        rpy_clear.close();  // 关闭文件流
        std::cout << "rpy File created successfully." << std::endl;
      } else {
        std::cout << "Failed to create rpy file." << std::endl;
      }

      std::ofstream sigma_clear(result_sigma,
                                std::ios::out | std::ios::trunc);  // 创建新文件
      if (sigma_clear.is_open()) {  // 检查是否成功打开
        sigma_clear << std::endl;
        sigma_clear.close();  // 关闭文件流
        std::cout << "sigma File created successfully." << std::endl;
      } else {
        std::cout << "Failed to create sigma file." << std::endl;
      }
    }

    image_folder_path = readParam<std::string>(nh, "image_folder_path");
    pcd_folder_path = readParam<std::string>(nh, "pcd_folder_path");

    boost::filesystem::directory_iterator image_end_itr, pcd_end_itr;
    std::cout << "start!!" << std::endl;

    for (boost::filesystem::directory_iterator i_itr(image_folder_path);
         i_itr != image_end_itr; ++i_itr) {
      if (boost::filesystem::is_regular_file(i_itr->path())) {
        image_files.push_back(i_itr->path().string());
      }
    }
    for (boost::filesystem::directory_iterator j_itr(pcd_folder_path);
         j_itr != pcd_end_itr; ++j_itr) {
      if (boost::filesystem::is_regular_file(j_itr->path())) {
        pcd_files.push_back(j_itr->path().string());
      }
    }
    sort(image_files.begin(), image_files.end());
    sort(pcd_files.begin(), pcd_files.end());

    while (!image_files.empty() && !pcd_files.empty()) {
      std::string im = image_files.front();
      std::string pcd = pcd_files.front();
      std::cout << im << std::endl << pcd << std::endl;

      image_h(im);
      pcl::PointCloud<pcl::PointXYZ> plane_in = pcd_h(pcd);

      if (runSolver(plane_in, pcd)) {  // im, pcd
        std::cout << "end end end" << std::endl;

        while (!image_files.empty() && !pcd_files.empty()) {
          l2c_project(image_files.back(), pcd_files.back());
          image_files.pop_back();
          pcd_files.pop_back();
        }
        break;
        // image_files.clear();
        // pcd_files.clear();
      } else {
        image_files.pop_front();
        pcd_files.pop_front();
      }
    }
  }

  void image_h(std::string &image_name) {
    std::cout << "start image_h" << std::endl;
    image_in = cv::imread(image_name, cv::IMREAD_COLOR);

    cv::Mat undistorted_img;

    cv::Mat RR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat mapx_open, mapy_open;

    cv::Mat intrinsic_undis;
    projection_matrix.copyTo(intrinsic_undis);
    cv::Mat distCoeff_4dim = cv::Mat(4, 1, cv::DataType<double>::type);
    distCoeff_4dim.at<double>(0) = 0.07253199815750122;
    distCoeff_4dim.at<double>(1) = -0.02428469993174076;
    distCoeff_4dim.at<double>(2) = -0.010476499795913696;
    distCoeff_4dim.at<double>(3) = 0.00593825988471508;
    // 鱼眼模型对图像校正
    cv::fisheye::initUndistortRectifyMap(projection_matrix, distCoeff_4dim, RR,
                                         intrinsic_undis, image_in.size(),
                                         CV_32FC1, mapx_open, mapy_open);

    cv::remap(image_in, undistorted_img, mapx_open, mapy_open,
              cv::INTER_LINEAR);

    std::cout << "image in" << std::endl;
    boardDetectedInCam = cv::findChessboardCorners(
        undistorted_img, cv::Size(checkerboard_cols, checkerboard_rows),
        image_points,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
    if (boardDetectedInCam) {
      std::cout << "board detect" << std::endl;
      // // 亚像素提高精度
      // cv::TermCriteria criteria = cv::TermCriteria(
      // cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);
      // cv::cornerSubPix(image_in, image_points, cv::Size(11, 11), cv::Size(-1,
      // -1), criteria);

    } else {
      std::cout << "board not detect   " << image_name << std::endl;
    }
    cv::drawChessboardCorners(undistorted_img,
                              cv::Size(checkerboard_cols, checkerboard_rows),
                              image_points, boardDetectedInCam);

    if (image_points.size() == object_points.size() &&
        image_points.size() > 0) {
      // cv::solvePnP(object_points, image_points, projection_matrix, distCoeff,
      // rvec, tvec, false, CV_ITERATIVE);

      // std::cout << "object_points:" << std::endl << object_points <<
      // std::endl;
      // std::cout << "image_points:" << std::endl << image_points <<
      // std::endl;
      cv::Mat i_dis = cv::Mat::zeros(5, 1, CV_64F);
      cv::solvePnP(object_points, image_points, projection_matrix, i_dis, rvec,
                   tvec, false);
      std::cout << "rvec:" << std::endl << rvec << std::endl;
      std::cout << "tvec:" << std::endl << tvec << std::endl;

      projected_points.clear();
      // cv::fisheye::projectPoints(object_points, projected_points, rvec, tvec,
      //                            projection_matrix, distCoeff_4dim);
      cv::projectPoints(object_points, rvec, tvec, projection_matrix, i_dis, projected_points, cv::noArray());
      for (int i = 0; i < projected_points.size(); i++) {
        cv::circle(undistorted_img, projected_points[i], 1, cv::Scalar(0, 0, 255), 2,
                   cv::LINE_AA, 0);
        boost::filesystem::path filePath(image_name);
        std::string image_filename_p = filePath.filename().string();
        cv::imwrite("/home/pw/Desktop/im/" + image_filename_p, undistorted_img);
      }
      // cv::cv2eigen(rvec, e_rvec);
      // cv::cv2eigen(tvec, e_tvec);

      // Eigen::Matrix3d c_R_v= d_cam_vehicle_tf.block<3, 3>(0, 0);
      // e_rvec_v = c_R_v.inverse() * e_rvec;
      // e_tvec_v = c_R_v.inverse() * e_tvec;

      // // std::cout << "e_rvec_v:" << std::endl << e_rvec_v << std::endl;
      // // std::cout << "e_tvec_v:" << std::endl << e_tvec_v << std::endl;

      // cv::eigen2cv(e_rvec_v, rvec);

      cv::Rodrigues(rvec, C_R_W);
      cv::cv2eigen(C_R_W, c_R_w);
      c_t_w = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1),
                              tvec.at<double>(2));

      r3 = c_R_w.block<3, 1>(0, 2);
      // Nc = (r3.dot(e_tvec_v)) * r3;
      Nc = (r3.dot(c_t_w)) * r3;
      // Nc = r3;
      // Nc = d_cam_vehicle_tf.block<3, 3>(0, 0).inverse() * Nc;
      std::cout << "pnp sloved" << std::endl;
    }
    cv::resize(undistorted_img, image_resized, cv::Size(), 0.25, 0.25);
    std::cout << "view board" << std::endl;
    cv::imshow("view", image_resized);
    cv::waitKey(100);
    std::cout << "end image_h" << std::endl;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr norm_plane_boundary(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_, std::string pcd_file_name_) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered_p(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_p;
    ne_p.setInputCloud(cloud_in_);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_p(new pcl::search::KdTree<pcl::PointXYZ>());
    ne_p.setSearchMethod(tree_p);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Use all neighbors in a sphere of radius 1cm
    // ne_p.setRadiusSearch(1);
    ne_p.setKSearch(20);
    ne_p.compute(*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud_in_, *normals, *cloud_with_normals);
    // （2）采用RANSAC提取平面
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
    // pcl::PCDWriter writer;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients(true);//设置对估计的模型系数需要进行优化
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE); //设置分割模型
    seg.setNormalDistanceWeight(1);//设置表面法线权重系数
    seg.setMethodType(pcl::SAC_RANSAC);//设置采用RANSAC作为算法的参数估计方法
    seg.setMaxIterations(500); //设置迭代的最大次数
    seg.setDistanceThreshold(0.5); //设置内点到模型的距离允许最大值
    seg.setInputCloud(cloud_in_);
    seg.setInputNormals(normals);
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
    // Extract the planar inliers from the input cloud
    extract.setInputCloud(cloud_in_);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);
    extract.filter(*plane_filtered_p);
    pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_file_name_
                              +"_pp.pcd", *plane_filtered_p);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_bd;
    ne_bd.setInputCloud(plane_filtered_p);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_bd(new pcl::search::KdTree<pcl::PointXYZ>());
    ne_bd.setSearchMethod(tree_bd);
    pcl::PointCloud<pcl::Normal>::Ptr normals_bd(new pcl::PointCloud<pcl::Normal>);
    // Use all neighbors in a sphere of radius 1cm
    // ne_bd.setRadiusSearch(1);
    ne_bd.setKSearch(20);
    ne_bd.compute(*normals_bd);                        
    //calculate boundary
    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points(
        new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZ> boundary_points;
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
    est.setInputCloud(plane_filtered_p);
    est.setInputNormals(normals_bd);
    est.setSearchMethod(tree_bd);
    est.setKSearch(50); //一般这里的数值越高，最终边界识别的精度越好
    est.compute(boundary);
    std::cout << "boundary" << std::endl << boundary << std::endl;
    
    // pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_res(new pcl::PointCloud<pcl::PointXYZ>);

    // 遍历pcl::Boundary中的每个点
    std::vector<int> boundary_indices;
    for (size_t i = 0; i < boundary.points.size(); ++i) {
        if (boundary.points[i].boundary_point) {
            boundary_indices.push_back(static_cast<int>(i));
        }
    }

    // 从原始点云中提取边界点并保存到新的点云
    pcl::copyPointCloud(*plane_filtered_p, boundary_indices, *boundary_points);
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    // sor.setInputCloud(boundary_points);
    // sor.setMeanK(15);
    // sor.setStddevMulThresh(1);
    // sor.filter(*boundary_points);

    pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_file_name_
                              +"_bd.pcd", *boundary_points);
    return boundary_points;

  }

  // void calculate_center(pcl::PointCloud<pcl::PointXYZ> boundary_points_) {
  //   pcl::PointXYZ minPt, maxPt;
  //   pcl::getMinMax3D(boundary_points_, minPt, maxPt);
	//   std::cerr << " " << minPt.x << " " << minPt.y << " " << minPt.z << std::endl;
	//   std::cerr << " " << maxPt.x << " " << maxPt.y << " " << maxPt.z << std::endl;
    // Eigen::Vecter3d delt;
//     std::vector<pcl::PointXYZ> v_point(6, pcl::PointXYZ);
//     std::vector<double> center_p(3, 0);
//     sort(boundary_points_.points.begin(), boundary_points_.points.end(), [](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
//         return p1.x < p2.x;
//     });
//     // pcl::PointXYZ x_min = boundary_points_.begin();
//     // pcl::PointXYZ x_max = boundary_points_.end();
//     pcl::PointXYZ x_min = *std::min_element(boundary_points_.points.begin(), boundary_points_.points.end(), [](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
//     return p1.x < p2.x;
// });

// pcl::PointXYZ x_max = *std::max_element(boundary_points_.points.begin(), boundary_points_.points.end(), [](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
//     return p1.x < p2.x;
// });
//     std::cout << x_min << "    " << x_max << std::endl;
//     double delt_x = x_max.x - x_min.x;
//     std::cout << delt_x << std::endl;

    // delt.push_back(delt_x);
    // v_point.push_back(boundary_points_.points.begin());
    // v_point.push_back(boundary_points_.points.end());
    // std::cout << delt_x << std::endl;
    

    // sort(boundary_points_.points.begin(), boundary_points_.points.end(), [](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
    //     return p1.y < p2.y;
    // });
    // pcl::PointCloud<pcl::PointXYZ> y_min = boundary_points_.begin();
    // pcl::PointCloud<pcl::PointXYZ> y_max = boundary_points_.end();
    // double delt_y = y_max->y - y_min->y;
    // delt.push_back(delt_y);
    // v_point.push_back(boundary_points_.points.begin());
    // v_point.push_back(boundary_points_.points.end());

    // sort(boundary_points_.points.begin(), boundary_points_.points.end(), [](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
    //     return p1.z < p2.z;
    // });
    // pcl::PointCloud<pcl::PointXYZ> z_min = boundary_points_.begin();
    // pcl::PointCloud<pcl::PointXYZ> z_max = boundary_points_.end();
    // double delt_x = z_max->z - z_min->z;
    // delt.push_back(delt_z);
    // v_point.push_back(boundary_points_.points.begin());
    // v_point.push_back(boundary_points_.points.end());

    // sort(delt.begin(), delt.end());
    // if(delt.begin() == delt_x) {
    //   center_p[0] = (v_point[2]->x + v_point[3]->x + v_point[4]->x + v_point[5]->x) / 4;
    //   center_p[1] = (v_point[2]->y + v_point[3]->y + v_point[4]->y + v_point[5]->y) / 4;
    //   center_p[2] = (v_point[2]->z + v_point[3]->z + v_point[4]->z + v_point[5]->z) / 4;
    // } else if(delt.begin() == delt_y) {
    //   center_p[0] = (v_point[0]->x + v_point[1]->x + v_point[4]->x + v_point[5]->x) / 4;
    //   center_p[1] = (v_point[0]->y + v_point[1]->y + v_point[4]->y + v_point[5]->y) / 4;
    //   center_p[2] = (v_point[0]->z + v_point[1]->z + v_point[4]->z + v_point[5]->z) / 4;
    // } else {
    //   center_p[0] = (v_point[0]->x + v_point[1]->x + v_point[2]->x + v_point[3]->x) / 4;
    //   center_p[1] = (v_point[0]->y + v_point[1]->y + v_point[2]->y + v_point[3]->y) / 4;
    //   center_p[2] = (v_point[0]->z + v_point[1]->z + v_point[2]->z + v_point[3]->z) / 4;
    // }
    // return center_p
  // }

  // 不好使
  void get_bd_lines(pcl::PointCloud<pcl::PointXYZ>::Ptr bd_cloud_in, std::string pcd_file_name_) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);  // 选择线模型
    seg.setMethodType(pcl::SAC_RANSAC);    // 使用RANSAC算法
    seg.setInputCloud(bd_cloud_in);      // 设置输入点云

    for (int i = 0; i < 4; ++i) {
        // 提取一条边
        pcl::ModelCoefficients::Ptr coefficients_bdline(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // 设置SAC分割参数
        seg.segment(*inliers, *coefficients_bdline);
        std::cout << "boundary_line" << std::endl << coefficients_bdline << std::endl;
    
        // 从外轮廓点云中提取该边的点
        pcl::PointCloud<pcl::PointXYZ>::Ptr edge(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(bd_cloud_in);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*edge);

        // 存储该边的点云
        *edge_cloud += *edge;    
    }
    pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_file_name_
                              +"_bd_line.pcd", *edge_cloud);

  }

  pcl::PointCloud<pcl::PointXYZ> pcd_h(std::string &pcd_name) {
    std::cout << "start pcd_h" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_name, *in_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered(
        new pcl::PointCloud<pcl::PointXYZ>);

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

    // filter ground point
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud_filtered_z);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr fil_ground_map(
        new pcl::PointCloud<pcl::PointXYZ>);
    ne.setSearchMethod(tree);
    ne.setKSearch(10);
    pcl::PointCloud<pcl::Normal>::Ptr f_normals(
        new pcl::PointCloud<pcl::Normal>);
    ne.compute(*f_normals);

    for (size_t i = 0; i < f_normals->size(); ++i) {
      if (f_normals->points[i].normal_z > 0.7) {
        continue;
      } else {
        pcl::PointXYZ pt;
        pt.x = cloud_filtered_z->points[i].x;
        pt.y = cloud_filtered_z->points[i].y;
        pt.z = cloud_filtered_z->points[i].z;
        fil_ground_map->push_back(pt);
      }
    }

    /// Plane Segmentation
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
        new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(fil_ground_map));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(ransac_threshold);
    ransac.computeModel();
    std::vector<int> inliers_indicies;
    ransac.getInliers(inliers_indicies);
    pcl::copyPointCloud<pcl::PointXYZ>(*fil_ground_map, inliers_indicies,
                                       *plane);

    /// Statistical Outlier Removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(plane);
    sor.setMeanK(100);
    sor.setStddevMulThresh(1);
    sor.filter(*plane_filtered);

    /// Store the points lying in the filtered plane in a vector
    lidar_points.clear();
    for (size_t i = 0; i < plane_filtered->points.size(); i++) {
      double X = plane_filtered->points[i].x;
      double Y = plane_filtered->points[i].y;
      double Z = plane_filtered->points[i].z;
      lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
    }

    // 将结果保存为新的PCD文件
    // std::string filted_pcd_name;
    // boost::split(std::string, pcd_name, boost::is_any_of("."))

    boost::filesystem::path filePath(pcd_name);
    std::string pcd_filename = filePath.filename().string();
    // pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_filename
    //                           +"_x.pcd", *cloud_filtered_x);
    // pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_filename
    //                           +"_y.pcd", *cloud_filtered_y);
    // pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_filename
    //                           +"_z.pcd", *cloud_filtered_z);
    // pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_filename
    //                           +"_ground_removed_cloud.pcd", *fil_ground_map);
    // pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_filename
    //                           +"_plane.pcd", *plane);
    // if(lidar_points.size() > min_points_on_plane) {
    //   pcl::io::savePCDFileASCII("/home/pw/Desktop/pcdpcd/" + pcd_filename
    //                           +"_plane_in.pcd", *plane_filtered);
    // }

    pcl::PointCloud<pcl::PointXYZ>::Ptr bd_cloud = norm_plane_boundary(plane, pcd_filename);
    // get_bd_lines(bd_cloud, pcd_filename);
    // std::vector<double> bd_center =  
    // calculate_center(*bd_cloud);
    // std::cout << "bd center" << std::endl;
    // std::cout << bd_center[0] << std::endl;
    // std::cout << bd_center[1] << std::endl;
    // std::cout << bd_center[2] << std::endl;

    return *plane_filtered;
    std::cout << "end pcd_h" << std::endl;
  }

  Eigen::Matrix4f ll_cc_p() {
    // 73相机-车 rpy   deg - rad
    float c_roll = 0.013 / 180 * M_PI;
    float c_pitch = 3.498 / 180 * M_PI;
    float c_yaw = 0.224 / 180 * M_PI;

    Eigen::Matrix3f cam_rotation_matrix;
    cam_rotation_matrix = Eigen::AngleAxisf(c_yaw, Eigen::Vector3f::UnitZ()) *
                          Eigen::AngleAxisf(c_pitch, Eigen::Vector3f::UnitY()) *
                          Eigen::AngleAxisf(c_roll, Eigen::Vector3f::UnitX());

    Eigen::Vector3f t_camera(1.854933, 0.003, 1.297982);

    Eigen::Matrix4f cam_transformation_matrix = Eigen::Matrix4f::Identity();

    cam_transformation_matrix.block<3, 3>(0, 0) = cam_rotation_matrix;  //
    cam_transformation_matrix.block<3, 1>(0, 3) = t_camera;

    // // 71相机-车 rpy-q
    // float c_roll = 1.053;
    // float c_pitch = 2.781;
    // float c_yaw = 0.091;

    // Eigen::Matrix3f cam_rotation_matrix;
    // cam_rotation_matrix = Eigen::AngleAxisf(c_roll, Eigen::Vector3f::UnitZ())
    //                     * Eigen::AngleAxisf(c_pitch,
    //                     Eigen::Vector3f::UnitY())
    //                     * Eigen::AngleAxisf(c_yaw, Eigen::Vector3f::UnitX());
    // Eigen::Vector3f t_camera(1.865162, -0.023901, 1.29708);

    // Eigen::Matrix4f cam_transformation_matrix = Eigen::Matrix4f::Identity();
    // cam_transformation_matrix.block<3, 3>(0, 0) = cam_rotation_matrix;
    // cam_transformation_matrix.block<3, 1>(0, 3) = t_camera;

    // left-lidar - 车
    float l_roll = 0.05046696 / 180 * M_PI;
    float l_pitch = 0.54040612 / 180 * M_PI;
    float l_yaw = 26.47351612 / 180 * M_PI;
    Eigen::Matrix3f li_rotation_matrix;
    li_rotation_matrix = Eigen::AngleAxisf(l_yaw, Eigen::Vector3f::UnitZ()) *
                         Eigen::AngleAxisf(l_pitch, Eigen::Vector3f::UnitY()) *
                         Eigen::AngleAxisf(l_roll, Eigen::Vector3f::UnitX());
    Eigen::Vector3f t_lidar(3.61345053, 0.7944809, 0.41564371);

    Eigen::Matrix4f li_transformation_matrix = Eigen::Matrix4f::Identity();
    li_transformation_matrix.block<3, 3>(0, 0) = li_rotation_matrix;
    li_transformation_matrix.block<3, 1>(0, 3) = t_lidar;

    // Eigen::Matrix4f L2C = cam_transformation_matrix *
    // li_transformation_matrix;
    Eigen::Matrix4f L2C =
        cam_transformation_matrix.inverse() * li_transformation_matrix;

    return L2C;  //
  }

  //
  void l2c_project(std::string &image_name, std::string &pcd_name) {
    std::cout << "start project" << std::endl;
    cv::Mat pre_image = cv::imread(image_name);

    cv::Mat undistorted_img;

    cv::Mat RR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat mapx_open, mapy_open;

    cv::Mat intrinsic_undis;
    projection_matrix.copyTo(intrinsic_undis);
    cv::Mat distCoeff_4dim = cv::Mat(4, 1, cv::DataType<double>::type);
    distCoeff_4dim.at<double>(0) = 0.07253199815750122;
    distCoeff_4dim.at<double>(1) = -0.02428469993174076;
    distCoeff_4dim.at<double>(2) = -0.010476499795913696;
    distCoeff_4dim.at<double>(3) = 0.00593825988471508;
    // cv::undistort(pre_image, undistorted_img, projection_matrix, distCoeff);

    cv::fisheye::initUndistortRectifyMap(projection_matrix, distCoeff_4dim, RR,
                                         intrinsic_undis, pre_image.size(),
                                         CV_32FC1, mapx_open, mapy_open);

    cv::remap(pre_image, undistorted_img, mapx_open, mapy_open,
              cv::INTER_LINEAR);

    // 加载点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr pre_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_name, *pre_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ccctransformed_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();  //
    // Eigen::Matrix4f C_T_L_floatMatrix = C_T_L_tf.cast<float>();

    Eigen::Matrix4f C_T_L_floatMatrix = C_T_L.cast<float>();

    // transform.block<3,3>(0, 0) = C_T_L.block(0, 0, 3, 3);  //    设置旋转矩阵
    // R transform.block<3,1>(0, 3) = C_T_L.block(0, 3, 3, 1);  // 设置平移向量
    // t

    // transform.block<3, 3>(0, 0) =
    //     C_T_L_floatMatrix.block<3, 3>(0, 0);  // 设置旋转矩阵 R
    // transform.block<3, 1>(0, 3) =
    //     C_T_L_floatMatrix.block<3, 1>(0, 3);  // 设置平移向量 t

    // Eigen::Matrix4f C_T_L_floatMatrix = Eigen::Matrix4f::Identity();
    // C_T_L_floatMatrix.block<3, 3>(0, 0) = C_T_L.cast<float>().block<3, 3>(0,
    // 0); C_T_L_floatMatrix.block<3, 1>(0, 3) = ll_cc_p().block<3, 1>(0, 3);

    std::cout << "start trans" << std::endl;
    pcl::transformPointCloud(*pre_cloud, *transformed_cloud, C_T_L_floatMatrix);
    Eigen::Matrix4f body_transform = f_cam_vehicle_tf * ll_cc_p();
    pcl::transformPointCloud(*pre_cloud, *ccctransformed_cloud, body_transform);

    std::cout << "transform " << std::endl;
    std::cout << C_T_L_floatMatrix << std::endl;
    std::cout << "ll_cc_p" << std::endl;
    std::cout << ll_cc_p() << std::endl;
    std::cout << "body_transform" << std::endl;
    std::cout << body_transform << std::endl;

    boost::filesystem::path filePath(pcd_name);
    std::string pcd_filename = filePath.filename().string();
    pcl::io::savePCDFileASCII(
        "/home/pw/Desktop/pcdpcd/" + pcd_filename + "_output.pcd",
        *transformed_cloud);
    // pcl::io::savePCDFileASCII(
    //     "/home/pw/cjy_home/calib_ws/src/cam_lidar_calib/output_2_73/"
    //     "transformed.pcd",
    //     *transformed_cloud);
    pcl::io::savePCDFileASCII(
        "/home/pw/Desktop/pcdpcd/" + pcd_filename + "_cccoutput.pcd",
        *ccctransformed_cloud);
    cv::Mat projected_image = undistorted_img.clone();  // undistorted_img

    cv::Mat projected_ccimage = undistorted_img.clone();  // undistorted_img

    // 遍历每个点并投影到相机图像
    std::cout << "start draw" << std::endl;

    for (const pcl::PointXYZ &point : transformed_cloud->points) {
      // 将点从相机坐标系投影到像素坐标系
      cv::Point2d pixel_point;
      pixel_point.x = (point.x * projection_matrix.at<double>(0, 0) / point.z) +
                      projection_matrix.at<double>(0, 2);
      pixel_point.y = (point.y * projection_matrix.at<double>(1, 1) / point.z) +
                      projection_matrix.at<double>(1, 2);

      // 检查点是否在图像范围内
      if (pixel_point.x >= 0 &&
          pixel_point.x < undistorted_img.cols &&  // undistorted_img
          pixel_point.y >= 0 &&
          pixel_point.y < undistorted_img.rows) {  // undistorted_img
        // 获取点的颜色（假设点云有颜色信息）
        cv::circle(projected_image, cv::Point(pixel_point.x, pixel_point.y), 2,
                   cv::Scalar(0, 0, 255), -1);
      }
    }
    // 保存并显示带有投影点的相机图像
    boost::filesystem::path filePath_i(image_name);
    std::string image_filename = filePath_i.filename().string();
    cv::imwrite("/home/pw/Desktop/im/" + pcd_filename + "im.png",
                projected_image);
    cv::resize(projected_image, projected_image, cv::Size(), 0.25, 0.25);
    cv::imshow("Projected Image", projected_image);
    while (true) {
      int key = cv::waitKey(10);
      if (key == 27)  // 按下ESC键，退出循环
        break;
    }

    for (const pcl::PointXYZ &point : ccctransformed_cloud->points) {
      // 将点从相机坐标系投影到像素坐标系
      cv::Point2d pixel_point_cc;
      pixel_point_cc.x =
          (point.x * projection_matrix.at<double>(0, 0) / point.z) +
          projection_matrix.at<double>(0, 2);
      pixel_point_cc.y =
          (point.y * projection_matrix.at<double>(1, 1) / point.z) +
          projection_matrix.at<double>(1, 2);

      // 检查点是否在图像范围内
      if (pixel_point_cc.x >= 0 &&
          pixel_point_cc.x < pre_image.cols &&  // undistorted_img
          pixel_point_cc.y >= 0 &&
          pixel_point_cc.y < pre_image.rows) {  // undistorted_img
        // 获取点的颜色（假设点云有颜色信息）
        cv::circle(projected_ccimage,
                   cv::Point(pixel_point_cc.x, pixel_point_cc.y), 2,
                   cv::Scalar(0, 0, 255), -1);
      }
    }
    // 保存并显示带有投影点的相机图像
    boost::filesystem::path filePath_ci(image_name);
    std::string image_filename_ci = filePath_ci.filename().string();
    cv::imwrite("/home/pw/Desktop/im/" + pcd_filename + "_ccim.png",
                projected_ccimage);
    cv::resize(projected_ccimage, projected_ccimage, cv::Size(), 0.25, 0.25);
    cv::imshow("Projected ccimage", projected_ccimage);
    while (true) {
      int key = cv::waitKey(10);
      if (key == 27)  // 按下ESC键，退出循环
        break;
    }
  }

  void readCameraParams(std::string cam_config_file_path, int &image_height,
                        int &image_width, cv::Mat &D, cv::Mat &K) {
    cv::FileStorage fs_cam_config(cam_config_file_path, cv::FileStorage::READ);
    if (!fs_cam_config.isOpened())
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
  T readParam(ros::NodeHandle &n, std::string name) {
    T ans;
    if (n.getParam(name, ans)) {
      ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    } else {
      ROS_ERROR_STREAM("Failed to load " << name);
      n.shutdown();
    }
    return ans;
  }

  void addGaussianNoise(Eigen::Matrix4d &transformation) {
    std::vector<double> data_rot = {0, 0, 0};
    const double mean_rot = 0.0;
    std::default_random_engine generator_rot;
    generator_rot.seed(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> dist(mean_rot, 90);
    // std::normal_distribution<double> dist(mean_rot, 5);

    // Add Gaussian noise
    for (auto &x : data_rot) {
      x = x + dist(generator_rot);
    }

    double roll = data_rot[0] * M_PI / 180;
    double pitch = data_rot[1] * M_PI / 180;
    double yaw = data_rot[2] * M_PI / 180;

    Eigen::Matrix3d m;
    m = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    std::vector<double> data_trans = {0, 0, 0};
    const double mean_trans = 0.0;
    std::default_random_engine generator_trans;
    generator_trans.seed(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> dist_trans(mean_trans, 0.5);

    // Add Gaussian noise
    for (auto &x : data_trans) {
      x = x + dist_trans(generator_trans);
    }

    Eigen::Vector3d trans;
    trans(0) = data_trans[0];
    trans(1) = data_trans[1];
    trans(2) = data_trans[2];

    Eigen::Matrix4d trans_noise = Eigen::Matrix4d::Identity();
    trans_noise.block(0, 0, 3, 3) = m;
    trans_noise.block(0, 3, 3, 1) = trans;
    transformation = transformation * trans_noise;
  }

  void rotation2rpy(Eigen::Matrix3d rotation_) {
    Eigen::Vector3d rpy_angles = rotation_.eulerAngles(2, 1, 0);
    float roll_degrees = rpy_angles(2) * 180.0 / M_PI;
    float pitch_degrees = rpy_angles(1) * 180.0 / M_PI;
    float yaw_degrees = rpy_angles(0) * 180.0 / M_PI;
    std::cout << "rotation_" << std::endl;
    std::cout << "Roll (X-axis): " << roll_degrees << " degrees"
              << std::endl;
    std::cout << "Pitch (Y-axis): " << pitch_degrees << " degrees"
              << std::endl;
    std::cout << "Yaw (Z-axis): " << yaw_degrees << " degrees"
              << std::endl;
  }

  bool runSolver(pcl::PointCloud<pcl::PointXYZ> plane_in_,
                 std::string &pcd_name) {  // std::string &image_name,
                                           // std::string &pcd_name
    std::cout << "start solve" << lidar_points.size() << std::endl;
    if (lidar_points.size() > min_points_on_plane && boardDetectedInCam) {
      // std::cout << "r3.dot(r3_old)  " << r3.dot(r3_old) << std::endl;
      if (r3.dot(r3_old) < 0.90) {
        boost::filesystem::path filePath(pcd_name);
        std::string pcd_filename = filePath.filename().string();
        pcl::io::savePCDFileASCII(
            "/home/pw/Desktop/pcdpcd/" + pcd_filename + "_plane_in.pcd",
            plane_in_);
        r3_old = r3;
        all_normals.push_back(Nc);
        all_lidar_points.push_back(lidar_points);
        ROS_ASSERT(all_normals.size() == all_lidar_points.size());
        std::cout << "Recording View number: " << all_normals.size()
                  << std::endl;
        // std::cout << "good view" << std::endl << image_name << std::endl <<
        // pcd_name << std::endl;;
        if (all_normals.size() >= num_views) {
          std::cout << "Starting optimization..." << std::endl;
          // init_file.open(initializations_file);

          for (int counter = 0; counter < no_of_initializations; counter++) {
            /// Start Optimization here
            std::cout << "solve num  " << counter + 1 << " of "
                      << no_of_initializations << std::endl;

            /// Step 1: Initialization
            // Eigen::Matrix4d transformation_matrix =
            // Eigen::Matrix4d::Identity(); Eigen::Matrix4d
            // transformation_matrix = llll_cc_p();
            // Eigen::Matrix4d transformation_matrix;
            Eigen::Matrix4d transformation_matrix = ll_cc_p().cast<double>();
            Eigen::Matrix4f temp_init = f_cam_vehicle_tf * ll_cc_p();
            transformation_matrix = temp_init.cast<double>();
            std::cout << "initial_transformation_matrix" << std::endl;
            std::cout << transformation_matrix << std::endl;

            // addGaussianNoise(transformation_matrix);
            Eigen::Matrix3d Rotn = transformation_matrix.block(0, 0, 3, 3);

            // 车上外参rpy
            Eigen::Vector3d rpy_angles = Rotn.eulerAngles(2, 1, 0);  // ZYX顺序
            // 输出RPY弧度
            std::cout << "Roll (X-axis): " << rpy_angles(2) << " radians"
                      << std::endl;
            std::cout << "Pitch (Y-axis): " << rpy_angles(1) << " radians"
                      << std::endl;
            std::cout << "Yaw (Z-axis): " << rpy_angles(0) << " radians"
                      << std::endl;
            // 角度
            double roll_degrees = rpy_angles(2) * 180.0 / M_PI;
            double pitch_degrees = rpy_angles(1) * 180.0 / M_PI;
            double yaw_degrees = rpy_angles(0) * 180.0 / M_PI;
            std::cout << "Roll (X-axis): " << roll_degrees << " degrees"
                      << std::endl;
            std::cout << "Pitch (Y-axis): " << pitch_degrees << " degrees"
                      << std::endl;
            std::cout << "Yaw (Z-axis): " << yaw_degrees << " degrees"
                      << std::endl;

            Eigen::Vector3d axis_angle;
            ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());

            Eigen::Vector3d Translation =
                transformation_matrix.block(0, 3, 3, 1);

            // Eigen::Vector3d rpy_init = Rotn.eulerAngles(2, 1, 0) * 180 /
            // M_PI; Eigen::Vector3d tran_init = transformation_matrix.block(0,
            // 3, 3, 1);

            // Eigen::Matrix3d ori_extrin;
            // ori_extrin << -0.0031189598549765194, -0.99998180471893194,  -0.0051635569610389634,
            //               -0.073016140628518703,  0.0053775315812755698, -0.99731626145461383,
            //               0.99732588219555351,  -0.0027335663809816197,  -0.073031584384438394;

            // Eigen::Matrix3d new_extrin_60;
            // new_extrin_60 << -0.0048849724283204247, -0.99996399680355907,   -0.0069384537920903757, 
            //                  -0.072123540102200359,   0.0072727833882243543, -0.99736919021238701,
            //                   0.99738374360498794,   -0.0043716951447226905, -0.072156470770546793;

            // Eigen::Matrix3d new_extrin_30;
            // new_extrin_30 << -0.0031140965054861961,  -0.9999615239138453,     -0.0082007984278611012,  
            //                  -0.072385723371097069,    0.0084047344696518648,  -0.99734129940083005,
            //                   0.99737185114429483,     0.0025121963288199053,  -0.07240911140600978; 
            
            // Eigen::Matrix3d new_extrin_20;
            // new_extrin_20 << -0.0038777804730477476, -0.99997494721075675,  -0.0059218045768185479,  
            //                  -0.073011464856193337,  0.0061891636457650414, -0.99731189717806812, 
            //                    0.99732356275090395, -0.003434996973669684, -0.07303363593328209; 
            // Eigen::Matrix3d new_extrin_40;
            // new_extrin_40 << -0.0031128197569130463, -0.99997109448092558, -0.0069368981381811631,  
            //                  -0.072133185949640771, 0.0071433942162859483,  -0.9973694277467221,
            //                   0.99739015126374453, -0.0026042506963159175,  -0.072153337001106349; 
            
            // Eigen::Matrix3d new_extrin_50;
            // new_extrin_50 << -0.0031166565604179939, -0.99996746576979489, -0.0074400170575971408,  
            //                   -0.073016154322009155, 0.0076477547811555574, -0.99730143540197025,
            //                   0.99732588839351222,  -0.0025650046277239402,  -0.073037614222746947;

            Eigen::Matrix3d new_extrin_301;
            new_extrin_301 << -0.0026131590155783731, -0.99981944495888975,  -0.018821500526151117,  
                              -0.073025742760984794, 0.018962106649893246,  -0.99714977781956238,
                              0.9973266327007243,  -0.0012312568759927089, -0.073062108594945024;
                       
            Eigen::Matrix3d new_extrin_302;
            new_extrin_302 << -0.0034983715978429077,  -0.99995816369029766, -0.0084517610763132529,  
                               -0.073013935505717859, 0.0086846762779283979,  -0.9972931071755754,
                              0.99732478492120624, -0.0028718055427318636,  -0.073041263096926684;

            Eigen::Matrix3d new_extrin_303;
            new_extrin_303 <<  -0.0040042543479638006, -0.99997865158543542,  -0.0051635569610389634,  
                              -0.073011351222034698, 0.0054421715265067496, -0.99731626145461372,
                              0.99732307129628217,  -0.0036165097053872455, -0.073031584384438381;

            Eigen::Matrix3d new_extrin_20;
            new_extrin_20 << -0.0034983705244152831, -0.99997634312721473,  -0.0059223803994602649,  
                              -0.073013807888232168, 0.0061620355388647347,  -0.99731189362991013,
                              0.99732479426781251,  -0.0030565509875964451, -0.073033637693366987;

            Eigen::Matrix3d new_extrin_90;
            new_extrin_90 <<  -0.0034924926404251259,  -0.99998055157763321, -0.0051671037968933353,  
                              0.071878844638523093, 0.0054048041979494104, -0.99739872658080553,
                              0.99740725593339608, -0.0031120022611048365, -0.071896322947277658;

            Eigen::Matrix3d new_extrin_50;
            new_extrin_50 << -0.0034983721203366985, -0.99998054934684444, -0.0051635569610389625,  
                              -0.07301409503307929, 0.0054052349339965902,  -0.99731626145461372,
                              0.99732477324039637,  -0.0031119709655692317, -0.073031584384438394;
                              
            rotation2rpy(new_extrin_301);
            rotation2rpy(new_extrin_302);
            rotation2rpy(new_extrin_303);
            rotation2rpy(new_extrin_20);
            rotation2rpy(new_extrin_90);
            rotation2rpy(new_extrin_50);

            Eigen::VectorXd R_t(6);
            // Eigen::VectorXd R_t(3);
            R_t(0) = axis_angle(0);
            R_t(1) = axis_angle(1);
            R_t(2) = axis_angle(2);

            R_t(3) = Translation(0);
            R_t(4) = Translation(1);
            R_t(5) = Translation(2);

            std::cout << "end solve initial" << std::endl;
            /// Step2: Defining the Loss function (Can be NULL)
            //  ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
            // ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::LossFunction *loss_function = NULL;

            /// Step 3: Form the Optimization Problem
            // fixed xyz
            // ceres::Problem problem;
            // problem.AddParameterBlock(R_t.data(), 3);
            // for (int i = 0; i < all_normals.size(); i++) {
            //   Eigen::Vector3d normal_i = all_normals[i];
            //   std::vector<Eigen::Vector3d> lidar_points_i = all_lidar_points[i];
            //   for (int j = 0; j < lidar_points_i.size(); j++) {
            //     Eigen::Vector3d lidar_point = lidar_points_i[j];
            //     ceres::CostFunction *cost_function =
            //         new ceres::AutoDiffCostFunction<CalibrationRotErrorTerm, 1,
            //                                         3>(
            //             new CalibrationRotErrorTerm(lidar_point, normal_i,
            //                                         Translation));
            //     problem.AddResidualBlock(cost_function, loss_function,
            //                              R_t.data());
            //   }
            // }

            ceres::Problem problem;
            problem.AddParameterBlock(R_t.data(), 6);
            for (int i = 0; i < all_normals.size(); i++) {
              Eigen::Vector3d normal_i = all_normals[i];
              std::vector<Eigen::Vector3d> lidar_points_i =
              all_lidar_points[i]; for (int j = 0; j < lidar_points_i.size();
              j++) {
                Eigen::Vector3d lidar_point = lidar_points_i[j];
                ceres::CostFunction *cost_function =
                    new ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1,
                    6>(
                        new CalibrationErrorTerm(lidar_point, normal_i));
                problem.AddResidualBlock(cost_function, loss_function,
                                         R_t.data());
              }
            }
            std::cout << "end Form the Optimization Problem" << std::endl;

            /// Step 4: Solve it
            ceres::Solver::Options options;
            options.max_num_iterations = 400;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_type = ceres::TRUST_REGION;  // 使用LM算法
            // options.ite
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            std::cout << "start solve" << std::endl;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << std::endl;
            std::cout << "end solve" << std::endl;
            // cv::destroyWindow("view");

            /// Printing and Storing C_T_L in a file
            ceres::AngleAxisToRotationMatrix(R_t.data(), Rotn.data());
            C_T_L.block(0, 0, 3, 3) = Rotn;
            C_T_L.block(0, 3, 3, 1) = Eigen::Vector3d(R_t[3], R_t[4], R_t[5]);
            // C_T_L.block(0, 3, 3, 1) = Translation;

            // Eigen::Matrix4d cam_vehicle_tf;
            // cam_vehicle_tf << 0, -1, 0, 0,
            //                   0, 0, -1, 0,
            //                   1, 0, 0, 0,
            //                   0, 0, 0, 1;

            C_T_L_tf = d_cam_vehicle_tf * C_T_L;
            std::cout << "RPY = " << std::endl
                      << Rotn.eulerAngles(2, 1, 0) * 180 / M_PI << std::endl;
            std::cout << "trans_res " << std::endl << C_T_L << std::endl;

            // std::cout << "t = " << C_T_L_tf.block(0, 3, 3, 1) << std::endl;
            std::cout << "t = " << std::endl
                      << C_T_L.block(0, 3, 3, 1) << std::endl;

            // /// Step 5: Covariance Estimation
            // std::cout << "start Covariance Estimation" << std::endl;
            // ceres::Covariance::Options options_cov;
            // ceres::Covariance covariance(options_cov);
            // std::vector<std::pair<const double *, const double *> >
            //     covariance_blocks;
            // covariance_blocks.push_back(std::make_pair(R_t.data(),
            // R_t.data())); CHECK(covariance.Compute(covariance_blocks,
            // &problem)); double covariance_xx[3 * 3];
            // covariance.GetCovarianceBlock(R_t.data(), R_t.data(),
            //                               covariance_xx);

            // Eigen::MatrixXd cov_mat_RotTrans(3, 3);
            // cv::Mat cov_mat_cv = cv::Mat(3, 3, CV_64F, &covariance_xx);
            // cv::cv2eigen(cov_mat_cv, cov_mat_RotTrans);

            // Eigen::MatrixXd cov_mat_TransRot(3, 3);
            // cov_mat_TransRot.block(0, 0, 3, 3) =
            //     cov_mat_RotTrans.block(3, 3, 3, 3);
            // cov_mat_TransRot.block(3, 3, 3, 3) =
            //     cov_mat_RotTrans.block(0, 0, 3, 3);
            // cov_mat_TransRot.block(0, 3, 3, 3) =
            //     cov_mat_RotTrans.block(3, 0, 3, 3);
            // cov_mat_TransRot.block(3, 0, 3, 3) =
            //     cov_mat_RotTrans.block(0, 3, 3, 3);

            // double sigma_xx = sqrt(cov_mat_TransRot(0, 0));
            // double sigma_yy = sqrt(cov_mat_TransRot(1, 1));
            // double sigma_zz = sqrt(cov_mat_TransRot(2, 2));

            // double sigma_rot_xx = sqrt(cov_mat_TransRot(3, 3));
            // double sigma_rot_yy = sqrt(cov_mat_TransRot(4, 4));
            // double sigma_rot_zz = sqrt(cov_mat_TransRot(5, 5));

            // std::cout << "sigma_xx = " << sigma_xx << "\t"
            //           << "sigma_yy = " << sigma_yy << "\t"
            //           << "sigma_zz = " << sigma_zz << std::endl;

            // std::cout << "sigma_rot_xx = " << sigma_rot_xx * 180 / M_PI <<
            // "\t"
            //           << "sigma_rot_yy = " << sigma_rot_yy * 180 / M_PI <<
            //           "\t"
            //           << "sigma_rot_zz = " << sigma_rot_zz * 180 / M_PI
            //           << std::endl;
            // std::cout << "end Covariance Estimation" << std::endl;

            std::ofstream results;
            results.open(result_str, std::ios::out | std::ios::app);
            results << C_T_L << "\n\n";
            results.close();

            std::ofstream results_rpy;
            results_rpy.open(result_rpy, std::ios::out | std::ios::app);
            results_rpy << Rotn.eulerAngles(2, 1, 0) * 180 / M_PI << "\n\n";
            results_rpy.close();

            // std::ofstream results_sigma;
            // results_sigma.open(result_sigma, std::ios::out | std::ios::app);
            // results_sigma << sigma_xx << " " << sigma_yy << " " << sigma_zz
            //               << "\n"
            //               << sigma_rot_xx << " " << sigma_rot_yy << " "
            //               << sigma_rot_zz << "\n\n";
            // results_sigma.close();

            std::cout << "No of initialization: " << counter << std::endl;
          }
          // ROS_WARN_STREAM("end solve all");
          std::cout << "end solve all" << std::endl;
          return true;
          // ros::shutdown();
        }
      } else {
        std::cout << "Not enough Rotation, view not recorded" << std::endl;
        return false;
      }
    } else {
      if (!boardDetectedInCam) {
        std::cout << "Checker-board not detected in Image." << std::endl;
        return false;
      } else {
        ROS_WARN_STREAM("Checker Board Detected in Image?: "
                        << boardDetectedInCam << "\t"
                        << "No of LiDAR pts: " << lidar_points.size()
                        << " (Check if this is less than threshold) ");
        return false;
      }
    }
  }
  
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "hozon_CameraLidarCalib_node");
  ros::NodeHandle nh("~");
  camLidarCalib cLC(nh);
  // ros::spin();
  return 0;
}