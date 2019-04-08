#ifndef MODELS_RESIDUAL_H
#define MODELS_RESIDUAL_H

// TODO: Blocks and Heads should be moved into an anonymous namespace (probably need to copy portions to a .cpp)
struct ConvBlockImpl : public torch::nn::Module {
  private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm norm;
  public:
    ConvBlockImpl(int64_t filters=256)
      : conv(torch::nn::Conv2dOptions(17, filters, /*kernel_size=*/3).stride(1).padding(1)),  // TODO: register module might be able to go here?
        norm(torch::nn::BatchNormOptions(filters)) {
      conv = register_module("conv", conv);
      norm = register_module("norm", norm);
    }

    torch::Tensor forward(torch::Tensor x) {
      x = conv(x);
      x = norm(x);
      x = torch::relu(x);
      return x;
    }
};
TORCH_MODULE(ConvBlock);

struct ResidualBlockImpl : public torch::nn::Module {
  private:
      torch::nn::Conv2d conv1;
      torch::nn::BatchNorm norm1;
      torch::nn::Conv2d conv2;
      torch::nn::BatchNorm norm2;
  public:
    ResidualBlockImpl(int64_t filters=256)
      : conv1(torch::nn::Conv2dOptions(filters, filters, /*kernel_size=*/3).stride(1).padding(1)),
        norm1(torch::nn::BatchNormOptions(filters)),
        conv2(torch::nn::Conv2dOptions(filters, filters, /*kernel_size=*/3).stride(1).padding(1)),
        norm2(torch::nn::BatchNormOptions(filters)) {
      conv1 = register_module("conv1", conv1);
      norm1 = register_module("norm1", norm1);
      conv2 = register_module("conv2", conv2);
      norm2 = register_module("norm2", norm2);
    }
    torch::Tensor forward(torch::Tensor x) {
      x = conv1(x);
      x = norm1(x);
      x = torch::relu(x);
      x = conv2(x);
      x = torch::relu(x);
      x = norm1(x);
      return x;
    }
};
TORCH_MODULE(ResidualBlock);

struct PolicyHeadImpl : public torch::nn::Module {
  private:
      torch::nn::Conv2d conv;
      torch::nn::BatchNorm norm;
      torch::nn::Linear fc;
      int64_t channels;
  public:
    PolicyHeadImpl(int64_t filters=256, int64_t channels=2)
      : conv(torch::nn::Conv2dOptions(filters, channels, /*kernel_size=*/1).stride(1)),
        norm(torch::nn::BatchNormOptions(channels)),
        fc(19 * 19 * channels, 19 * 19 + 1),
        channels(channels) {
      conv = register_module("conv", conv);
      norm = register_module("norm", norm);
      fc = register_module("fc", fc);
    }
    torch::Tensor forward(torch::Tensor x) {
      x = conv(x);
      x = norm(x);
      x = torch::relu(x);
      x = x.reshape({x.size(0), -1});
      x = fc->forward(x);
      return x;
    }
};
TORCH_MODULE(PolicyHead);

struct DualNetworkImpl : torch::nn::Module {
private:
  ConvBlock conv;
  torch::nn::Sequential layer;
  PolicyHead policy;
public:
  DualNetworkImpl(int64_t layers, int64_t filters=256, int64_t policy_channels=2, int64_t value_hidden_dim=256)
      : conv(filters),
        layer(),
        policy(filters, policy_channels) {
    for (int64_t i = 0; i < layers; i++) {
      layer->push_back(ResidualBlock(filters));
    }
    conv = register_module("conv", conv);
    layer = register_module("layer", layer);
    policy = register_module("policy", policy);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv(x);
    x = layer->forward(x);
    x = policy(x);
    return x;
  }
};
TORCH_MODULE(DualNetwork);

#endif
