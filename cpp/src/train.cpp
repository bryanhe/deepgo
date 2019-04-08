#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include "datasets/kgs.hpp"
#include "models/residual.hpp"

const char *KGS_ROOT = "/home/bryanhe/deepgo/data/kgs/";
const size_t TRAIN_BATCH_SIZE = 128;
const size_t EPOCHS = 10;

int main(int argc, char *argv[]) {


  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  DualNetwork model;
  model.to(device);


  auto train_dataset = KGS(KGS_ROOT);
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), TRAIN_BATCH_SIZE);

  //                         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
  //                         .map(torch::data::transforms::Stack<>());
  //const size_t train_dataset_size = train_dataset.size().value();
  //auto train_loader =
  //    torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
  //        std::move(train_dataset), kTrainBatchSize);
  //
  //torch::optim::SGD optimizer(
  //    model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

//  for (size_t epoch = 1; epoch <= EPOCHS; epoch++) {
//      model.train();
//      size_t batch_idx = 0;
//      for (auto& batch : data_loader) {
//        auto data = batch.data.to(device), targets = batch.target.to(device);
//        optimizer.zero_grad();
//        auto output = model.forward(data);
//        auto loss = torch::nll_loss(output, targets);
//        AT_ASSERT(!std::isnan(loss.template item<float>()));
//        loss.backward();
//        optimizer.step();
//
//        //if (batch_idx++ % kLogInterval == 0) {
//          std::printf(
//              "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
//              epoch,
//              batch_idx * batch.data.size(0),
//              dataset_size,
//              loss.template item<float>());
//        //}
//      }
//  }

  return 0;

}
