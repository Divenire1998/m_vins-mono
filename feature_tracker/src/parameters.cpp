/*
 * @Copyright: 
 * @file name: File name
 * @Data: Do not edit
 * @LastEditor: 
 * @LastData: 
 * @Describe: 
 */
/*
 * @Copyright:
 * @file name: File name
 * @Data: Do not edit
 * @LastEditor:
 * @LastData:
 * @Describe:
 */
#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

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

// 读配置参数，通过roslaunch文件的参数服务器获得
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    // 从ROS服务器中获得配置文件的路径
    config_file = readParam<std::string>(n, "config_file");
    // 使用opencv的yaml文件接口来读取文件
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    // 读取参数
    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];

    // 特征点最大个数
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];


    // 发布给后端处理的帧频率
    FREQ = fsSettings["freq"];
    if (FREQ == 0)
        FREQ = 100;

    // ransac算法的门限
    F_THRESHOLD = fsSettings["F_threshold"];
    // 是否发布跟踪点的图像
    SHOW_TRACK = fsSettings["show_track"];
    // 是否均衡化
    EQUALIZE = fsSettings["equalize"];

    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";

    CAM_NAMES.push_back(config_file);

    // 连续被WINDOW_SIZE帧给观测到，就是红色的特征点
    // TODO 稳定？ 后端是关键帧还是普通帧判断稳定得？
    WINDOW_SIZE = 10;
    STEREO_TRACK = false;

    // 虚拟相机的焦距
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    fsSettings.release();
}
