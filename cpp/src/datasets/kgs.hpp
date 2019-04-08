#ifndef DATASETS_KGS_HPP
#define DATASETS_KGS_HPP
#include <torch/data/datasets/mnist.h>

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>



#include <torch/data/example.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>


#include <string>

class KGS : public torch::data::datasets::Dataset<KGS> {
public:
  enum class Mode { TRAIN, TEST };

  explicit KGS(const std::string &root, Mode mode=Mode::TRAIN);

  // Returns the `Example` at the given `index`.
  torch::data::Example<> get(size_t index) override;

  // Returns the size of the dataset.
  at::optional<size_t> size() const override;

};
#endif /* DATASETS_KGS_HPP */
