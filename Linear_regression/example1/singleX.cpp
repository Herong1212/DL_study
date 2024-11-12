#include <torch/torch.h>
#include <iostream>
#include <vector>

int main()
{
    // 准备数据
    std::vector<float> areas = {1500, 2000, 2500, 3000, 3500};            // 面积
    std::vector<float> prices = {300000, 400000, 500000, 600000, 700000}; // 房价

    // 转换数据为Tensor，确保类型为Float
    auto X = torch::tensor(areas, torch::kFloat).view({-1, 1});
    auto y = torch::tensor(prices, torch::kFloat).view({-1, 1});

    // 定义线性回归模型
    struct LinearRegression : torch::nn::Module
    {
        torch::nn::Linear fc{nullptr};

        LinearRegression()
        {
            fc = register_module("fc", torch::nn::Linear(1, 1)); // 输入特征1，输出特征1
        }

        torch::Tensor forward(torch::Tensor x)
        {
            return fc(x);
        }
    };

    auto model = std::make_shared<LinearRegression>();

    // 设置优化器，调整学习率
    // 学习率设置过高可能会导致训练过程中的数值不稳定，从而出现nan。尝试将学习率降低到 0.001 或更低的值
    torch::optim::SGD optimizer(model->parameters(), /*学习率*/ 0.001);

    // 训练模型
    for (size_t epoch = 1; epoch <= 1000; ++epoch)
    {
        // 前向传播
        auto output = model->forward(X);
        
        // 打印调试信息
        std::cout << "Output: " << output << std::endl;
        std::cout << "Target: " << y << std::endl;

        // 计算损失
        auto loss = torch::mse_loss(output, y);

        // 优化器步骤
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // 打印调试信息
        if (epoch % 100 == 0)
        {
            std::cout << "Epoch [" << epoch << "], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // 预测
    auto area_new = torch::tensor({4000}).view({-1, 1});
    auto predicted_price = model->forward(area_new).item<float>();

    std::cout << "预测的房价(4000平方英尺): " << predicted_price << " 美元" << std::endl;

    return 0;
}
