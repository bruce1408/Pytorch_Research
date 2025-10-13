// C++/LibTorch 侧
#include <torch/script.h>

int main() {
  torch::jit::script::Module m = torch::jit::load("resnet18.ts");
  at::Tensor x = torch::randn({1,3,224,224});
  at::Tensor y = m.forward({x}).toTensor();
}
