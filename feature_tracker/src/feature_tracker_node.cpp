#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

//每个相机都有一个FeatureTracker实例，即trackerData[i]
FeatureTracker trackerData[NUM_OF_CAM];

double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0; //上一帧相机的时间戳
bool init_pub = 0;

/**
 * @brief   ROS的回调函数，对新来的图像进行特征点追踪，发布
 * @Description readImage()函数对新来的图像使用光流法进行特征点跟踪
 *              追踪的特征点封装成feature_points发布到pub_img的话题下，
 *              图像封装成ptr发布在pub_match下
 * @param[in]   img_msg 输入的图像
 * @return      void
 */
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{

    // 第一张图象
    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }

    // detect unstable camera stream
    // 检查时间戳是否正常，这里认为超过一秒或者错乱就异常
    // 图像时间差太多光流追踪就会失败，这里没有描述子匹配，因此对时间戳要求就高
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        // 一些常规的reset操作
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag); // 告诉其他模块要重启了
        return;
    }

    last_image_time = img_msg->header.stamp.toSec();

    // frequency control
    // 控制一下发给后端的频率，模拟的局部定时器
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 这段时间的频率和预设频率十分接近，就认为这段时间很棒，重启一下，避免delta t太大
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    // 把ros message格式的图像转成cv::Mat
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        
    cv::Mat show_img = ptr->image;

    TicToc t_r;
    // 处理图像数据
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        // 单目
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        // 双目
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        // 显示去畸变后的图像，用于验证去畸变算法的正确性
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    //更新特征点的全局ID号
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            // 对于单目相机
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    // 1、将特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)，
    //封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img;
    // 2、将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
    if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;         // 去畸变的归一化相机坐标系
            auto &cur_pts = trackerData[i].cur_pts;           // 像素坐标
            auto &ids = trackerData[i].ids;                   // id
            auto &pts_velocity = trackerData[i].pts_velocity; // 归一化坐标下的速度

            // 只发布追踪大于1的，因为等于1没法构成重投影约束，也没法三角化
            // 也就是当前帧新提取的特征点是不会发布的
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on frist image
        // 发布图像得特征点相关信息
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        // 显示一下发送给后端处理得光流跟踪信息
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            // cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                // 对于当前帧得每一个特征点
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    //显示追踪状态，越红越好，越蓝越不行
                    // len = [0~1]
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);

                    cv::circle(tmp_img,
                               trackerData[i].cur_pts[j],
                               2,                                         // 半径
                               cv::Scalar(255 * (1 - len), 0, 255 * len), // 颜色
                               2);                                        // 厚度

                    // draw speed line
                    Vector2d tmp_cur_un_pts(trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity(trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    
                    // 向后端发布得频率是10Hz，也就是0.1s 换算成特征点得运动距离 （归一化平面）
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - tmp_pts_velocity/FREQ;
                    tmp_prev_un_pts.z() = 1;

                    Vector2d tmp_prev_uv;
                    // 投影到像素平面上
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255, 0, 0), 1, 8, 0);

                    // 绘制特征点的ID号
                    char name[10];
                    sprintf(name, "%d", trackerData[i].ids[j]);
                    cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            cv::imshow("vis", stereo_img);
            cv::waitKey(3);

            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    // ros初始化和设置句柄
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // 从config.yaml配置文件中读取参数
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        // 获得每个相机的内参
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // 如果是鱼眼相机，带个fisheye_mask
    if (FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if (!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    //订阅话题IMAGE_TOPIC(/cam0/image_raw),执行回调函数img_callback
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart", 1000);

    // 使用OPENCV 绘制光流跟踪情况
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    ros::spin();
    return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?