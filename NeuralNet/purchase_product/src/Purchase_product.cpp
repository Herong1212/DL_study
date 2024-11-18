#include "../include/Purchase_product.h"

/**
 * @brief 假设我们有以下特征：
 * 			1、用户访问次数（次数越多，购买可能性越大）。
 * 			2、用户停留时间（时间越长，购买可能性越大）。
 * 		  目标是通过这些输入，预测用户是否会购买产品（0表示不会，1表示会）。
 * @param visits_Number  	用户访问次数
 * @param residence_Time  	用户停留时间
 *
 * @return possibility 是否会购买产品
 */

Purchase::Purchase()
{
    // ps：初始化每一层并注册
	// 初始化隐藏层（2个输入 ---> 3个隐藏）
	hidden = register_module("hidden", torch::nn::Linear(2, 3));
	// 初始化输出层（3个隐藏层 ---> 1个输出）
	output = register_module("output", torch::nn::Linear(3, 1));
}

torch::Tensor Purchase::forward(torch::Tensor x)
{
	x = torch::relu(hidden->forward(x));	// 隐藏层 + ReLU
	x = torch::sigmoid(output->forward(x)); // 输出层 + Sigmoid
	return x;
}