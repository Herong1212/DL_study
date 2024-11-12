#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <torch/torch.h>

class NeuralNetwork : public torch::nn::Module
{
private:
    torch::nn::Linear fc1{nullptr}; // 第一层
    torch::nn::Linear fc2{nullptr}; // 第二层

public:
    NeuralNetwork(/* args */);
    torch::Tensor forward(torch::Tensor x);
};

#endif // NEURALNETWORK_H