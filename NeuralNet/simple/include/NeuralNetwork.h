#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <torch/torch.h>

class NeuralNetwork : public torch::nn::Module
{
public:
    NeuralNetwork(/* args */);
    torch::Tensor forward(torch::Tensor x); // 前向传播函数

private:
    torch::nn::Linear fc1{nullptr}; // 第一层
    torch::nn::Linear fc2{nullptr}; // 第二层
};

#endif // NEURALNETWORK_H