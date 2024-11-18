#ifndef PURCHASE_PRODUCT_H
#define PURCHASE_PRODUCT_H

#include <torch/torch.h>

// 定义一个简单的两层神经网络
class Purchase : public torch::nn::Module
{
public:
	Purchase();
	// 前向传播函数
	torch::Tensor forward(torch::Tensor x);

private:
	// ? 为什么用 {}？
	torch::nn::Linear hidden{nullptr}; // 隐藏层
	torch::nn::Linear output{nullptr}; // 输出层
};

#endif // PURCHASE_PRODUCT_H