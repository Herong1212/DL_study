#include "./include/NeuralNetwork.h"
#include <iostream>

int main()
{
    // step1：准备数据
    torch::Tensor X = torch::tensor({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}, torch::kFloat);
    torch::Tensor Y = torch::tensor({0.0, 1.0, 0.0, 1.0}, torch::kFloat).view({-1, 1});

    // step2：实例化模型
    NeuralNetwork model;

    // step3：定义损失函数和优化器
    torch::optim::SGD optimazer(model.parameters(), 0.01);

    // step4：训练模型
    for (size_t epoch = 1; epoch < 1000; epoch++)
    {
        optimazer.zero_grad();                              // 清空梯度
        auto output = model.forward(X);                     // 前向传播
        auto loss = torch::binary_cross_entropy(output, Y); // 计算损失
        loss.backward();                                    // 反向传播
        optimazer.step();                                   // 更新权重

        // 打印损失
        if (epoch % 100 == 0)
        {
            std::cout << "Epoch [" << epoch << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // step5：测试模型
    torch::Tensor new_input = torch::tensor({0.5, 0.5}, torch::kFloat);
    auto prediction = model.forward(new_input);
    std::cout << "Prediction for input [0.5, 0.5]: " << prediction.item<float>() << std::endl;

    return 0;
}
