#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
	// step1：初始化特征提取器
	auto featureExtractor = std::make_shared<FeatureExtractor>();

	// step2：读取两张图片
	cv::Mat image1 = cv::imread("../data/1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat image2 = cv::imread("../data/2.png", cv::IMREAD_GRAYSCALE);

	if (image1.empty() || image2.empty())
	{
		std::cerr << "Failed to load images!" << std::endl;
		return -1;
	}

	// step3：转换为张量格式
	auto tensor1 = torch::from_blob(image1.data, {1, 1, image1.rows, image1.cols}, torch::kUInt8).to(torch::kFloat) / 255.0;
	auto tensor2 = torch::from_blob(image2.data, {1, 1, image2.rows, image2.cols}, torch::kUInt8).to(torch::kFloat) / 255.0;
	std::cout << "----------------------------------------------" << std::endl;

	// step4：提取特征图
	auto featureMap1 = featureExtractor->forward(tensor1);
	std::cout << "Feature map stats for Image 1 - Max: "
			  << featureMap1.max().item<float>() << ", Min: "
			  << featureMap1.min().item<float>() << std::endl;
	std::cout << "Feature map size for Image 1: " << featureMap1.sizes() << std::endl;

	auto featureMap2 = featureExtractor->forward(tensor2);
	std::cout << "Feature map stats for Image 2 - Max: "
			  << featureMap2.max().item<float>() << ", Min: "
			  << featureMap2.min().item<float>() << std::endl;
	std::cout << "----------------------------------------------" << std::endl;

	// step5：检测关键点
	float threshold = 0.075; // 设置较低阈值
	auto keypoints1 = featureExtractor->detectKeypoints(featureMap1, threshold);
	std::cout << "Image 1 detected keypoints: " << keypoints1.size() << std::endl;

	if (keypoints1.empty())
		std::cerr << "No keypoints detected even with threshold: " << threshold << std::endl;

	auto keypoints2 = featureExtractor->detectKeypoints(featureMap2, threshold);
	std::cout << "Image 2 detected keypoints: " << keypoints2.size() << std::endl;
	std::cout << "----------------------------------------------" << std::endl;

	// step6：提取描述子
	auto descriptors1 = featureExtractor->extractDescriptors(featureMap1, keypoints1);
	
	if (descriptors1.size(0) == 0)
	{
		std::cerr << "No descriptors extracted for Image 1. Skipping..." << std::endl;
		return -1;
	}
	auto descriptors2 = featureExtractor->extractDescriptors(featureMap2, keypoints2);
	std::cout << "----------------------------------------------" << std::endl;

	// step7：特征匹配
	cv::BFMatcher matcher(cv::NORM_L2, true);
	std::vector<cv::DMatch> matches;
	matcher.match(cv::Mat(descriptors1.size(0), descriptors1.size(1), CV_32F, descriptors1.data_ptr()),
				  cv::Mat(descriptors2.size(0), descriptors2.size(1), CV_32F, descriptors2.data_ptr()), matches);

	// step8：显示匹配结果
	cv::Mat output;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output);
	cv::imshow("Matches", output);
	cv::waitKey(0);

	return 0;
}
