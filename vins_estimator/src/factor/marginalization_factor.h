/*
 * @Copyright:
 * @file name: File name
 * @Data: Do not edit
 * @LastEditor:
 * @LastData:
 * @Describe:
 */
#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

// 保存待marg的变量与相关联的变量之间的约束关系，存储待边缘化变量相关的残差项
// 待marg变量索引 drop_set
// 其中Evaluate是边缘化中求解残差和雅克比
struct ResidualBlockInfo
{
    // 残差因子
    ResidualBlockInfo(
        ceres::CostFunction *_cost_function,
        ceres::LossFunction *_loss_function,     // 核函数
        std::vector<double *> _parameter_blocks, //优化变量
        std::vector<int> _drop_set               //待marg的序号
        )
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set)
    {
    }

    // 待边缘化的各个残差块计算残差和雅克比矩阵，同时处理核函数的case
    void Evaluate();

    ceres::CostFunction *cost_function; 
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    // 雅克比
    double **raw_jacobians;

    // C++的数组是行优先的，所以Eigen这儿用行优先，方便转换数组
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;

    // 误差项
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; // global size
    std::unordered_map<long, int> parameter_block_idx;  // local size
};

class MarginalizationInfo
{
public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;                // 所有因子
    int m, n;                                                // m为要边缘化的变量个数，n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size;      // global size    地址->参与边缘化的变量的维度
    std::unordered_map<long, int> parameter_block_idx;       // local size     地址->参数边缘化的变量在Hx=b中x的索引，边缘化的变量排在前面
    std::unordered_map<long, double *> parameter_block_data; // 地址->参数块实际内容的地址

    int sum_block_size;

    std::vector<int> keep_block_size; // global size
    std::vector<int> keep_block_idx;  // local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;

    const double eps = 1e-8;
};

// 由于边缘化的costfuntion不是固定大小的，因此只能继承最基本的类
class MarginalizationFactor : public ceres::CostFunction
{
public:
    MarginalizationFactor(MarginalizationInfo *_marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo *marginalization_info;
};
