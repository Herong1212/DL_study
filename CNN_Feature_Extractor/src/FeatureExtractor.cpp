#include "FeatureExtractor.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// 构造函数：初始化 CNN 层
FeatureExtractor::FeatureExtractor()
{
	conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)));
	conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1)));
}

// 前向传播：提取特征图
torch::Tensor FeatureExtractor::forward(const torch::Tensor &x)
{
	auto out = torch::relu(conv1->forward(x));
	out = torch::max_pool2d(out, 2); // 最大池化
	out = torch::relu(conv2->forward(out));
	return out;
}

// 从特征图中检测关键点
std::vector<cv::KeyPoint> FeatureExtractor::detectKeypoints(const torch::Tensor &featureMap, float threshold)
{
	std::vector<cv::KeyPoint> keypoints;

	auto featureMapCPU = featureMap.squeeze(0).detach().cpu(); // 转换到 CPU

	auto data = featureMapCPU.accessor<float, 3>(); // 访问特征图数据

	for (int i = 0; i < featureMap.size(2); ++i)
	{
		for (int j = 0; j < featureMap.size(3); ++j)
		{
			if (data[0][i][j] > threshold)
			{													  // 根据阈值选择响应点
				keypoints.emplace_back(cv::KeyPoint(j, i, 1.0f)); // (x, y, size)
			}
		}
	}

	std::cout << "Detected keypoints: " << keypoints.size() << std::endl;
	return keypoints; // ! 如果未找到任何关键点，则返回空的 keypoints
}

// 提取描述子：在关键点位置获取局部特征
torch::Tensor FeatureExtractor::extractDescriptors(const torch::Tensor &featureMap, const std::vector<cv::KeyPoint> &keypoints)
{
	if (keypoints.empty())
	{
		std::cerr << "Error: No keypoints detected. Skipping descriptor extraction." << std::endl;
		return torch::empty({0}); // 返回空张量
	}

	std::vector<torch::Tensor> descriptors;
	auto featureMapCPU = featureMap.squeeze(0).detach().cpu(); // 转换到 CPU

	for (const auto &kp : keypoints)
	{
		int x = static_cast<int>(kp.pt.x);
		int y = static_cast<int>(kp.pt.y);

		// 添加边界检查
		if (x < 0 || x >= featureMapCPU.size(2) || y < 0 || y >= featureMapCPU.size(1))
		{
			std::cerr << "Keypoint (" << x << ", " << y << ") is out of bounds for feature map with size: " << featureMapCPU.sizes() << std::endl;
			continue; // 跳过越界的关键点
		}

		descriptors.push_back(featureMapCPU.index({0, y, x}));
	}

	if (descriptors.empty())
	{
		std::cerr << "No valid descriptors extracted. Returning empty tensor." << std::endl;
		return torch::empty({0});
	}

	// 将描述子堆叠为张量
	return torch::stack(descriptors); // ! 如果 descriptors 为空，则会触发错误
}
