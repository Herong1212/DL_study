#include <torch/torch.h> // PyTorch的C++ API的主要头文件，提供了【所有】深度学习相关的功能
#include <iostream>

int main()
{
    // step1：准备数据
    // 输入数据：X包含每个样本的年龄和年收入。
    // ? 注意这里：输入的特征（年龄和薪水）数值差异较大，一个是几十，一个是几万，可能导致模型在训练时无法有效学习。
    // X 是一个 4 行 2 列的二维张量。每行表示一个样本，每列是该样本的一个特征。
    torch::Tensor X = torch::tensor({{25.0, 50000.0}, {30.0, 60000.0}, {35.0, 70000.0}, {40.0, 80000.0}}, torch::kFloat);
    // 对比：cv::Mat X
    // y 是一个 4 行 1 列的二维张量。.view({-1, 1}) 将 y 转换成了一个列向量，形状为 (4, 1)。每一个值对应 X 的【答案】
    torch::Tensor y = torch::tensor({0.0, 0.0, 1.0, 1.0}, torch::kFloat).view({-1, 1}); // 标签 reshape 成列向量

    // 为防止输入的两个特征相差过大，将其进行标准化（或归一化）：
    X = (X - X.mean(0)) / X.std(0); // X.mean(0)计算每列的均值，X.std(0)计算每列的标准差。

    // step2：定义逻辑回归模型
    // 模型结构：使用线性层torch::nn::Linear进行输入与输出的映射，并用torch::sigmoid进行激活。
    struct LogisticRegression : torch::nn::Module
    {
        // 在模型中定义一个线性层linear，初始化为nullptr。
        torch::nn::Linear linear{nullptr}; // torch::nn::Linear用于定义线性变换。

        LogisticRegression()
        {
            // 注册线性层linear，输入特征数为2（年龄和收入），输出特征数为1（预测结果）。
            // register_module方法将线性层添加到模型中，以便PyTorch可以跟踪其参数。
            linear = register_module("linear", torch::nn::Linear(2, 1)); // 两个输入特征，一个输出
        }

        // 定义前向传播函数forward，输入参数为张量x，表示模型的输入数据。
        torch::Tensor forward(torch::Tensor x)
        {
            // 计算线性层的输出并通过torch::sigmoid激活函数将其转化为0到1之间的值，表示预测概率。
            return torch::sigmoid(linear->forward(x)); // 使用 Sigmoid 激活函数
        }
    };

    // 创建逻辑回归模型的实例model，并使用std::make_shared智能指针管理其内存。
    auto model = std::make_shared<LogisticRegression>();

    // step3：设置损失函数和优化器
    // 优化器：使用 SGD 优化器来更新参数。
    torch::optim::SGD optimizer(model->parameters(), /*学习率*/ 0.001); // ? 0.001 的时候一直是 0！

    // step4：训练模型
    // 1000：训练周期，epoch是当前训练周期的索引。
    for (size_t epoch = 1; epoch <= 2000; ++epoch)
    {
        // step4.1：前向传播
        auto output = model->forward(X);
        // 损失函数：使用二元交叉熵损失函数 torch::binary_cross_entropy() 进行二分类损失计算。
        auto loss = torch::binary_cross_entropy(output, y);

        // step4.2：反向传播和优化
        // 清空旧的梯度记录；
        optimizer.zero_grad(); // 将所有模型参数的梯度清零。由于PyTorch会累加梯度，因此在每次反向传播之前需要手动清零，以防止前一次计算的梯度影响本次梯度更新。
        // 计算新梯度；
        loss.backward(); // 通过反向传播计算模型所有参数的梯度，即损失函数相对于每个参数的偏导数。框架会自动遍历计算图并计算这些梯度。
        //  更新模型参数；
        optimizer.step(); // 根据已经计算好的梯度更新模型的参数。优化器会根据其定义的优化算法（例如SGD、Adam等）来调整模型参数。

        // 打印损失，条件判断：每100个周期打印一次损失值。
        if (epoch % 100 == 0)
        {
            // 打印当前周期和对应的损失值。loss.item<float>()将损失值从张量转换为浮点数
            std::cout << "Epoch [" << epoch << "], Loss: " << loss.item<float>() << std::endl;

            // 打印模型的权重和对应的梯度，方便调试。
            // std::cout << "Weights: " << model->linear->weight << std::endl;
            // std::cout << "Gradients: " << model->linear->weight.grad() << std::endl;
        }
    }

    // 测试模型
    // 预测：训练后，代码用一个新的输入进行预测，输出值接近1表示预测会购买，接近0表示不购买。
    torch::Tensor new_input = torch::tensor({{28.0, 55000.0}}, torch::kFloat);
    auto prediction = model->forward(new_input);
    std::cout << "Prediction for input [28, 55000]: " << prediction.item<float>() << std::endl;

    return 0;
}
