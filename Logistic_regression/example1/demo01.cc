#include <torch/torch.h>
#include <iostream>

int main()
{
    // step1：准备数据
    torch::Tensor X = torch::tensor({{1.0, 2.0}, {2.0, 3.0}, {3.0, 5.0}, {4.0, 7.0}}, torch::kFloat);
    torch::Tensor y = torch::tensor({0.0, 0.0, 1.0, 1.0}, torch::kFloat).view({-1, 1}); // 标签 reshape 成列向量

    // step2：定义逻辑回归模型
    struct LogisticRegression : torch::nn::Module
    {
        // 使用了一个torch::nn::Linear层，输出用 torch::sigmoid 激活，以将结果映射到 [0, 1] 之间。
        torch::nn::Linear linear{nullptr};

        LogisticRegression()
        {
            linear = register_module("linear", torch::nn::Linear(2, 1)); // 两个输入特征，一个输出
        }

        torch::Tensor forward(torch::Tensor x)
        {
            return torch::sigmoid(linear->forward(x)); // 使用 Sigmoid 激活函数
        }
    };

    auto model = std::make_shared<LogisticRegression>();

    // step3：设置损失函数和优化器 --- 使用SGD优化器来更新模型参数。
    torch::optim::SGD optimizer(model->parameters(), /*学习率*/ 0.01);

    // step4：训练模型
    for (size_t epoch = 1; epoch <= 1000; ++epoch)
    {
        // 前向传播
        auto output = model->forward(X);
        auto loss = torch::binary_cross_entropy(output, y); // 用于二分类任务

        // 反向传播和优化
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // 打印损失
        if (epoch % 100 == 0)
        {
            std::cout << "Epoch [" << epoch << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // step5：测试模型
    torch::Tensor new_input = torch::tensor({{5.0, 8.0}}, torch::kFloat);
    auto prediction = model->forward(new_input);
    std::cout << "Prediction for input [5.0, 8.0]: " << prediction.item<float>() << std::endl;

    // 运行此代码后，训练过程中会每隔100个epoch打印一次损失值。最后会输出对新输入数据的预测值。
    return 0;
}
