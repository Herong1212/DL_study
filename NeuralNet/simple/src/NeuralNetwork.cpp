#include "../include/NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
    // 初始化每一层并注册
    fc1 = register_module("fc1", torch::nn::Linear(2, 5)); // 输入层 -> 隐藏层
    fc2 = register_module("fc2", torch::nn::Linear(5, 1)); // 隐藏层 -> 输出层
}

torch::Tensor NeuralNetwork::forward(torch::Tensor x)
{
    x = torch::relu(fc1->forward(x));    // 使用 ReLU 激活函数
    x = torch::sigmoid(fc2->forward(x)); // 使用 Sigmoid 激活函数
    return x;
}