#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main() {
    // Load the TorchScript model
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("model.pt");
    
    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 10}));

    // Execute the model and turn its output into a tensor
    at::Tensor output = module->forward(inputs).toTensor();

    std::cout << output << std::endl;
}