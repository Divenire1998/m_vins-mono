#include "marginalization_factor.h"

/**
 * @brief 待边缘化的各个残差块计算残差和雅克比矩阵，同时处理核函数的case
 *
 */
void ResidualBlockInfo::Evaluate()
{
    // 确定残差的维数
    residuals.resize(cost_function->num_residuals());

    // 相关的参数块的数目，以及每个参数块的维度
    // 比如预积分残差涉及到4个参数块，
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();

    // ceres接口都是double数组，因此这里给雅克比准备数组
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    // 这里就是把jacobians每个matrix地址赋给raw_jacobians，然后把raw_jacobians传递给ceres的接口，这样计算结果直接放进了这个matrix
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        // 雅克比矩阵大小 残差×变量
        // 比如说预积分残差相对于位姿的雅克比就是 15 * 7
        // num_residuals ：15
        // block_sizes[i] ：7
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);

        // 直接内存赋值
        // ! 都是行优先，算是一种加速
        raw_jacobians[i] = jacobians[i].data();
        // 指向同一块内存
        // dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }

    // 调用各自重载的接口计算残差和雅克比
    // 这里实际上结果放在了jacobians
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    // std::vector<int> tmp_idx(block_sizes.size());
    // Eigen::MatrixXd tmp(dim, dim);
    // for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //     int size_i = localSize(block_sizes[i]);
    //     Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //     for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //     {
    //         int size_j = localSize(block_sizes[j]);
    //         Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //         tmp_idx[j] = sub_idx;
    //         tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //     }
    // }
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    // std::cout << saes.eigenvalues() << std::endl;
    // ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    // 见Ceres的官网关于鲁棒核的理论部分
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        // printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        // 一阶导
        double sqrt_rho1_ = sqrt(rho[1]);

        //
        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            // rho[2] 肯定是小于0的 执行这个函数
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            // 乘上一个权重
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    // ROS_WARN("release marginlizationinfo");

    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
}

/**
 * @brief 添加残差块相关信息（优化变量，待边缘化变量）
 * @param[in] residual_block_info
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    // 待marg的变量相关的残差信息保存一下
    factors.emplace_back(residual_block_info);

    // 残差因子设计到的参数块信息
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;

    // 与残差因子相关的各个参数块的大小
    // 以IMU预积分残差为例子，相关联的变量为
    // para_Pose[0],              7
    // para_SpeedBias[0],         9
    // para_Pose[1],              7
    // para_SpeedBias[1]          9
    // parameter_block_sizes 存储 7 9 7 9
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    // 保存与残差相关的变量的信息
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        // 把残差相关变量的地址以及对应的变量大小取出来，存在parameter_block_size中
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    // 待边缘化的参数块的信息
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        // 把待marg变量的地址取出来，丢到parameter_block_idx 占个位置
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

/**
 * @brief 计算每个和待边缘化有关变量的残差，对应的Jacobian，并将各参数块数据拷贝到统一的内存
 */
void MarginalizationInfo::preMarginalize()
{

    // 与待marg变量有关的残差
    for (auto it : factors)
    {
        // 每个因子的残差大小以及相对个各个参数块的参数和雅克比
        it->Evaluate();

        // 残差各个参数块的大小
        // 以IMU预积分残差为例子，相关联的变量为
        // para_Pose[0],              7
        // para_SpeedBias[0],         9
        // para_Pose[1],              7
        // para_SpeedBias[1]          9
        // parameter_block_sizes 存储 7 9 7 9
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();

        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            // 把参数块的地址和残差拿出来
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];

            // 把各个参数块都备份起来，使用map避免重复参数块，之所以备份，是为了后面的状态保留
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);

                // 计算每个因子对应的变量、误差项、残差雅克比矩阵 放到parameter_block_data中
                // 以预积分残差为例子
                // parameter_block_data 里保存了指向 para_Pose 或者 para_SpeedBias的指针
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

/**
 * @brief 分线程构造Ax = b
 *
 * @param[in] threadsstruct
 * @return void*
 */
void *ThreadsConstructA(void *threadsstruct)
{
    ThreadsStruct *p = ((ThreadsStruct *)threadsstruct);

    // 对于每一个因子
    for (auto it : p->sub_factors)
    {

        // 以预积分约束为例子，残差的维度为15，与4个参数块有关，
        // pose_i  7
        // pose_j 7
        // speedBias_i 9
        // speedBias_j 9
        // parameter_blocks.size() = 4
        // jacobian_i就是 15*9 或者 15*7的
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            // 取出该参数块的索引和大小
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];

            // 确保是local size 也就是李代数
            if (size_i == 7)
                size_i = 6;

            // 之前ResidualBlockInfo::Evaluate()函数中计算雅克比参数已经算好了各个残差和雅克比，这里取出来
            // it->jacobians[i] 就是 因子对于参数块 i 的雅克比 15*7 实际上左6维就可以了
            // 比如预积分因子it->jacobians[0] 就是对pose_0 的雅克比
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);

            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;

                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);

                // 两个雅克比都取出来了，向大的H矩阵中填充元素就好了
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }

            // 然后构建g矩阵
            // ? : 为什么不是-JTe
            // 理论上是-JTe,但是后面分解为先验残差的时候也是用没加负号的公式
            // 就像个互相相反的bug叠加了一下.
            // 最终分解出来的残差是一样的
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

/**
 * @brief: 对于所有与待marg变量相关的元素，多线程构建出信息矩阵，然后边缘化。
 *         从边缘化后的H 和 b 中反解一个 先验雅克比和先验残差
 * @return {*}
 */
void MarginalizationInfo::marginalize()
{
    int pos = 0;

    // =============Step 1 所有与待marg变量有关的参数块取出来，设置其在信息矩阵中的位置

    // 待marg参数
    for (auto &it : parameter_block_idx)
    {
        // value设置为索引值 ，这也就把需要边缘化的变量排在前面
        it.second = pos;
        // 确保信息矩阵中位姿的表示为李代数
        pos += localSize(parameter_block_size[it.first]);
    }

    // 总共待边缘化的参数块总大小（不是个数）
    m = pos;

    // 除了待marg变量外的其他参数
    for (const auto &it : parameter_block_size)
    {

        // 加入到边缘化数组的后面，就是不marg的变量
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            // 这样每个参数块的大小都能被正确找到
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    // 其他参数块的总大小
    n = pos - m;

    // ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    // =============Step 2  根据残差和残差的雅克比构建，构建信息矩阵

    // 构建AX=B
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    // 单个线程构建
    /*
    TicToc t_summing;
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */

    // multi thread
    // 往A矩阵和b矩阵中填东西，利用多线程加速
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;

    // 遍历每一个残差
    for (auto it : factors)
    {
        // 每个线程均匀分配任务
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }

    // 每个线程构造一个A矩阵和b矩阵，最后大家加起来
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        // 所以A矩阵和b矩阵大小一样，预设都是0
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        // 大小 和 索引
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;

        int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void *)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }

    for (int i = NUM_THREADS - 1; i >= 0; i--)
    {
        pthread_join(tids[i], NULL);
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    // ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    // ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

    /*代码这里求Amm的逆矩阵时，为了保证数值稳定性，做了Amm=1/2*(Amm+Amm^T)的运算，
    Amm本身是一个对称矩阵，所以  等式成立。接着对Amm进行了特征值分解,再求逆，更加的快速*/
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    // ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    // printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    // 设x_{m}为要被marg掉的状态量，x_{r}是与x_{m}相关的状态量，所以在最后我们要保存的是x_{r}的信息
    //
    //      |      |    |          |   |
    //      |  Amm | Amr|  m       |bmm|        |x_{m}|
    //  A = |______|____|      b = |__ |       A|x_{r}| = b
    //      |  Arm | Arr|  n       |brr|
    //      |      |    |          |   |
    //舒尔补
    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);

    //这里的A和b是marg过的A和b
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    //
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A); //求更新后 A特征值
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));
    Eigen::VectorXd S_sqrt = S.cwiseSqrt(); //矩阵开方得到雅克比 J. 意思是sqrt(S)
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    //线性化x0处的Jacobian 和残差
    //对Amm进行了特征值分解,再求逆，更加的快速,x.asDiagonal是diag(x)对角阵
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    // std::cout << A << std::endl
    //           << std::endl;
    // std::cout << linearized_jacobians << std::endl;
    // printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //       (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{

    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        // 保存一下没有被mag的参数
        // 这些参数受到被marg参数块的约束
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }

    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

/**
 * @brief Construct a new Marginalization Factor:: Marginalization Factor object
 * 边缘化信息结果的构造函数，根据边缘化信息确定参数块总数和大小以及残差维数
 *
 * @param[in] _marginalization_info
 */
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) : marginalization_info(_marginalization_info)
{
    int cnt = 0;

    // keep_block_size表示上一次边缘化留下来的参数块的大小
    // 也就是上一次被marg掉得变量，对那些参数块有约束
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }

    // 上一次边缘化留下来的残差维度
    // 虚拟残差，用b向量分解而来
    // printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};

/**
 * @brief 边缘化结果残差和雅克比的计算
 *
 * @param[in] parameters
 * @param[in] residuals
 * @param[in] jacobians
 * @return true
 * @return false
 */
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    // for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //     //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //     //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    // printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    // printf("residual %x\n", reinterpret_cast<long>(residuals));
    // }

    // 上一次边缘化保留的残差块的local size的和,也就是残差维数
    int n = marginalization_info->n;
    // 上次边缘化的被margin的残差块总和
    int m = marginalization_info->m;
    // 用来存储残差
    Eigen::VectorXd dx(n);

    // 遍历所有的剩下的有约束的残差块
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        // 取出当前参数块的值
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        // 取出之前参数块的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        // 如果不是流形，直接做差
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        // 如果是SE3
        else
        {
            // 平移直接做差
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();

            // 旋转李代数做差
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();

            // 确保实部大于0
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }

    // 更新残差　边缘化后的先验误差 e = e0 + J * dx
    // 个人理解：根据FEJ．雅克比保持不变，但是残差随着优化会变化，因此下面不更新雅克比　只更新残差
    // 可以参考　https://blog.csdn.net/weixin_41394379/article/details/89975386
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                // 把jacobian转换给global size 
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
