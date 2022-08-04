#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

//窗口中被成功跟踪的特征数量
int FeatureManager::getFeatureCount()
{
    int cnt = 0;

    // 对于滑动窗口内的所有特征点
    for (auto &it : feature)
    {
        // 当前特征被跟踪到的次数
        it.used_num = it.feature_per_frame.size();

        // 这个特征点的被观测次数大于2就认为是跟踪上了
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   1. 将路标点与滑动窗口内的其他路标点建立数据关联，若为已知点，则加到共视关系中；若为新点，则新建一个特征点。
 *          2. 计算次新帧与次次新帧的视差，根据视察大小决定次新帧是否为关键帧
 * @param[in]   frame_count 窗口内帧的个数
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：次新帧是关键帧;false：非关键帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    // 输入到后端的图像特征点数量
    ROS_DEBUG("input feature: %d", (int)image.size());

    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;

    // test
    int num = 0;
    // 遍历当前图像帧的每一个特征点
    for (auto &id_pts : image)
    {
        // test
        // cout<<++num<<"id"<<id_pts.first<<"\n";

        // 特征点的ID
        int feature_id = id_pts.first;

        // 一个路标点在一张图像中的坐标
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        //寻找滑动窗口内，这个特征点是否已经存在
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          { return it.feature_id == feature_id; });

        //如果没有则新建一个，并添加这图像帧
        if (it == feature.end())
        {
            // 在特征点管理器中，新创建一个特征点id，这里的frame_count就是该特征点在滑窗中的当前位置，作为这个特征点的起始位置
            feature.push_back(FeaturePerId(feature_id, frame_count));
            //
            feature.back().feature_per_frame.push_back(f_per_fra);
        }

        // 如果这是一个已有的特征点，就在对应的“组织”下增加一个帧属性
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    // 前两帧都设置为KF，追踪过少也认为是KF
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 把滑动窗口内的所有特征点取出来
    for (auto &it_per_id : feature)
    {

        // it_per_id 是FeaturePerId类型的 存储观测到当前特征点的所有图像

        // 计算的实际上是frame_count-1,也就是前一帧是否为关键帧
        // 因此起始帧至少得是frame_count - 2,同时至少覆盖到frame_count - 1帧
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // 计算特征点在上一帧和上上一帧的视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // 次新帧和次次新帧没有共视的路标点
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf pixels", parallax_sum / parallax_num * FOCAL_LENGTH);
        // 看看平均视差是否超过一个阈值（归一化平面上）
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ", j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}



//得到frame_count_l与frame_count_r两帧之间的对应特征点
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        // 保证需要的特征点被这两帧都观察到
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;

            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        // ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);

        //    如果特征点的深度小于0
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        // 逆深度
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历每一个特征点
    for (auto &it_per_id : feature)
    {
        // 特征点被观测到的次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        // 对比较稳定的特征点进行三角化
        // TODO 是被关键帧看到的次数还是
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 代表已经三角化过了
        if (it_per_id.estimated_depth > 0)
            continue;

        // imu_i 第一个观测到该特征点的帧ID
        int imu_i = it_per_id.start_frame;
        auto imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        //【R0 | t0】为第i帧相机坐标系到世界坐标系的变换矩阵
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // 遍历所有看到这个特征点的KF
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            // 得到该KF的相机坐标系位姿
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];

            // T_w_cj -> T_c0_cj
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            // T_c0_cj -> T_cj_c0相当于把c0当作世界系
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 构建超定方程的其中两个方程
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            // 看到这个特征点的KF包括当前帧
            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        //对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        // it_per_id->estimated_depth = -b / A;
        // it_per_id->estimated_depth = INIT_DEPTH;

        // it_per_id->estimated_depth = svd_V[2] / svd_V[3];
        it_per_id.estimated_depth = svd_method;

        // 具体太近就设置成默认值
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

// 移除外点
void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

/**
 * @brief  边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
 *
 * @param[in] marg_R  被移除的位姿
 * @param[in] marg_P
 * @param[in] new_R    转接地图点的位姿
 * @param[in] new_P
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果不是被移除的帧观测到，那么该地图点对应的起始帧id减一
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // 取出归一化相机坐标系坐标
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            // 该点不再被原来的第一帧看到，因此从中移除
            it->feature_per_frame.erase(it->feature_per_frame.begin());

            // 如果这个地图点没有至少被两帧看到，删去
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            // 地图点管辖权移交
            else
            {
                // 看看地图点能不能被下一帧看到，如果能的话，转移一下深度值
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

/**
 * @brief 这个还没初始化结束，因此相比刚才，不进行地图点新的深度的换算，因为此时还有进行视觉惯性对齐
 */
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果特征点起始帧号start_frame不为零则减一
        if (it->start_frame != 0)
            it->start_frame--;
        //如果start_frame为0则直接移除feature_per_frame的第0帧FeaturePerFrame
        //如果feature_per_frame为空则直接删除特征点
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//边缘化次新帧时，对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果地图点被最后一帧看到，由于滑窗，他的起始帧减一
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            //如果次新帧之前已经跟踪结束则什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;

            int j = WINDOW_SIZE - 1 - it->start_frame;

            //如果在次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            //如果feature_per_frame为空则直接删除特征点
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);

            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * @brief 计算某个特征点it_per_id在次新帧和次次新帧的视差（归一化平面上）
 * @param {FeaturePerId} &it_per_id
 * @param {int} frame_count
 * @return {*}
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    // frame_i 上一帧
    // frame_j 上上一帧
    // 计算
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    //
    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;

    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;

    // 归一化相机坐标系的坐标差
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;

    // 当都是归一化坐标系时，他们两个都是一样的
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 返回视差
    // ! 注意这儿不是视差交
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}