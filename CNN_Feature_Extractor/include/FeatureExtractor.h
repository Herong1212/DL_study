#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

class FeatureExtractor : public torch::nn::Module
{
public:
	FeatureExtractor();																							   // 构造函数
	torch::Tensor forward(const torch::Tensor &x);																   // 提取特征的前向传播
	std::vector<cv::KeyPoint> detectKeypoints(const torch::Tensor &featureMap, float threshold = 0.1);			   // 从特征图中检测关键点
	torch::Tensor extractDescriptors(const torch::Tensor &featureMap, const std::vector<cv::KeyPoint> &keypoints); // 提取描述子

private:
	torch::nn::Conv2d conv1{nullptr}; // 第一层卷积
	torch::nn::Conv2d conv2{nullptr}; // 第二层卷积
};

#endif
