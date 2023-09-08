#include "data_loader.h"
#include <experimental/filesystem>
#include <glog/logging.h>

namespace DataLoader
{
    DataLoader():image_ptr(new ImageDequeType()),\
                            pcd_ptr(new CloudDequeType()),\
                            times_ptr(new TimeDequeType())
    {
        last_file_pose = 0;
        haveReadAllImage = false;
        haveReadAllLidar = false;
    }
    bool haveReadAllFile(){
        return haveReadAllImage && haveReadAllLidar;
    }
    ~DataLoader(){};

    void setImageBufferPtr(ImuDequePtr &imageDqPtr){
        image_ptr = imageDqPtr;
    }
    void setLidarBufferPtr(CloudDequePtr &cloudDqPtr){
        clouds_ptr = cloudDqPtr;
    }
    void setTimeBufferPtr(TimeDequePtr &timeDqPtr){
        times_ptr = timeDqPtr;
    }

    bool string_contains_pcd(const string& str){
        return str.find(".pcd") != string::npos;
    }
    bool string_contains_image(const string& str){
        return str.find(".jpg") != string::npos;
    }

    bool file_exists(const string& filename){
        ifstream file(filename);
        return file.good();
    }

    void getLidarFileName(){
        lidarFileDeque.clear();
        namespace fs = std::experimental::filesystem;
        for (const auto & entry : fs::directory_iterator(lidarFileDirPath)){
            std::string file_full_path = lidarFileDirPath + '/' +  entry.path().filename().string();
            if(!string_contains_pcd(file_full_path)){
                file_full_path += '/' + entry.path().filename().string() + ".pcd";
            }
            if(file_exists(file_full_path)){
                lidarFileDeque.push_back(file_full_path);
            }
            else{
                LOG(ERROR) << "pcd_file: "<< file_full_path << " donn't exists.";
            }
            // times_ptr->push_back(std::stoull(entry.path().filename().string())/1e6);
        }
        std::sort(lidarFileDeque.begin(),lidarFileDeque.end());
        scanSize = lidarFileDeque.size();
        if(scanSize == 0){
            LOG(WARNING) << "no pcd files founded";
        }
    }

    void getImageFileName(){
        imageFileDeque.clear();
        namespace fs = std::experimental::filesystem;
        for (const auto & entry : fs::directory_iterator(imageFileDirPath)){
            std::string file_full_path = imageFileDirPath + '/' +  entry.path().filename().string();
            if(!string_contains_image(file_full_path)){
                file_full_path += '/' + entry.path().filename().string() + ".jpg";
            }
            if(file_exists(file_full_path)){
                imageFileDeque.push_back(file_full_path);
            }
            else{
                LOG(ERROR) << "image_file: "<< file_full_path << " donn't exists.";
            }
            // times_ptr->push_back(std::stoull(entry.path().filename().string())/1e6);
        }
        std::sort(imageFileDeque.begin(),imageFileDeque.end());
        scanSize = imageFileDeque.size();
        if(scanSize == 0){
            LOG(WARNING) << "no image files founded";
        }
    }

    void setFilePath(std::string image_folder_path,std::string pcd_folder_path){
        imageFilePath = image_folder_path;
        lidarFileDirPath = pcd_folder_path;
        LOG(INFO) << "IMAGE File Path: " << imageFilePath\
                << "\tLidar Directory: " << lidarDir;
        getLidarFileName();
        getImageFileName();
    }

    PointCloudHuawei DataOP::readHuaweiCloud(){
        PointCloudHuawei cloud;
        if(clouds_ptr->size() == 0){
            haveReadAllLidar = true;
            return cloud;
        }
        cloud = clouds_ptr->front();
        clouds_ptr->pop_front();
        return cloud;        
    }

    PointCloudRS128 DataOP::readRS128Cloud(){
        PointCloudRS128 cloud;
        if(lidarFileDeque.size() == 0){
            haveReadAllLidar = true;
            return cloud;
        }
        std::string path = lidarFileDeque.front();
        pcl::PCDReader reader;
        reader.read(path,cloud);
        cloud.header.stamp= getTimeFromString(path);
        lidarFileDeque.pop_front();
        return cloud;
    }

    


} // namespace DataLoader
