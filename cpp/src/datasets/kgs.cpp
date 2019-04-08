#include <experimental/filesystem>
#include "kgs.hpp"

KGS::KGS(const std::string &root, Mode mode) {
  return;
  for (const auto &entry : std::experimental::filesystem::recursive_directory_iterator(root)) {
    std::cout << entry.path() << std::endl;
  }
}

torch::data::Example<> KGS::get(size_t index) {
  return {at::zeros({17, 19, 19}), at::zeros({1})};
  // return {at::zeros({17, 19, 19}), at::zeros({1}), at::zeros({1})};
  // return {images_[index], targets_[index]};
}

at::optional<size_t> KGS::size() const {
  return 1;
}
