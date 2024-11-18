#include "Purchase_product.h"
#include <iostream>
#include <fstream>

int main()
{
	// 准备模拟数据
	torch::Tensor data = torch::tensor({{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}});
	std::cout << "data shape: " << data.sizes() << std::endl;

	// 测试用
	// std::ofstream outfile("output.txt");
	for (int i = 0; i < data.size(0); i++)
	{
		// std::cout << data[i].item<float>() << std::endl;
		// 在 Python 中，print(tensor) 会以矩阵形式直接输出，因为 Python 内置了对张量的更强大支持。
		// 在 C++ 中，由于 std::cout 和 std::ofstream 是标准流，它们无法理解张量的高层次格式，必须手动处理。
		std::cout << data[i] << std::endl;
		// outfile << data[i] << std::endl;
	}
	// outfile.close();

	torch::Tensor labels = torch::tensor({{0.0}, {0.0}, {1.0}, {1.0}});
	std::cout << "labels shape: " << labels.sizes() << std::endl;

	// 创建一个神经网络
	Purchase model;
	// auto model2 = std::make_shared<Purchase>();

	// 损失函数和优化器
	torch::optim::SGD optimizer(model.parameters(), 0.01);
	// torch::optim::SGD optimizer(model2->parameters(), 0.01);
	// 二分类的交叉熵损失函数
	torch::nn::BCELoss loss_fn;

	// note：训练模型
	for (size_t epoch = 0; epoch < 1000; epoch++)
	{
		// model2->train();

		optimizer.zero_grad();

		// 前向传播
		auto output = model.forward(data);
		// auto output2 = model2->forward(data);

		// 计算损失
		auto loss = loss_fn(output, labels);
		// auto loss = loss_fn(output2, labels);

		// 反向传播
		loss.backward();
		optimizer.step();

		// 打印损失
		if (epoch % 100 == 0)
		{
			std::cout << "Epoch = [" << loss.item<float>() << "]" << std::endl;
		}
	}

	// 测试模型
	// model2->eval();
	auto test_data = torch::tensor({{1.5, 2.5}, {3.5, 4.5}});
	std::cout << "test_data shape: " << test_data.sizes() << std::endl;
	auto predictions = model.forward(test_data);
	std::cout << "Prediction shape: " << predictions.sizes() << std::endl;

	// auto predictions = model2->forward(test_data);
	// std::cout << "Prediction: \n"
	// 		  << predictions.item<float>() << std::endl;
	for (int i = 0; i < predictions.size(0); ++i)
		std::cout << predictions[i].item<float>() << "\n";

	return 0;
}

/*
	如果需要改进或扩展
	1、调整超参数：
		例如修改学习率、批大小等，可以观察是否有更好的收敛速度或更低的最终损失。
	2、添加更多训练数据：
		如果数据规模较小，可以尝试添加更多样本，模型可能会有更好的泛化能力。
	3、验证和测试：
		使用一个单独的验证集，测试模型的性能是否在未见过的数据上表现良好。
	4、改进网络结构：
	如果想进一步提升模型的表现，可以尝试增加隐藏层的神经元数量或添加更多隐藏层。
*/