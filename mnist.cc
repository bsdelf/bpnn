#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "bpnn.h"
using namespace sml;

static inline int b2l(int num) {
  return ((num >> 24) & 0xff) |
         ((num << 8) & 0xff0000) |
         ((num >> 8) & 0xff00) |
         ((num << 24) & 0xff000000);
}

/*
template <typename L>
static inline auto argmax(const L& l) -> typename L::difference_type {
    const auto& iter = std::max_element(l.begin(), l.end());
    return std::distance(l.begin(), iter);
}
*/

template <typename num_t>
num_t str2num(const std::string& str) {
  std::stringstream stream;
  stream << str;
  num_t num;
  stream >> std::dec >> num;
  return num;
}

template <typename L>
auto argmax(const L& l) -> int {
  const auto& iter = std::max_element(l.begin(), l.end());
  return std::distance(l.begin(), iter);
}

std::vector<int> rand_seq(int max) {
  std::vector<int> lst;
  lst.reserve(max);
  for (decltype(max) v = 0; v < max; ++v) {
    lst.push_back(v);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, max - 1);
  for (size_t i = 0; i < lst.size(); ++i) {
    size_t o = (i + dis(gen)) % (max - 1);
    std::swap(lst[i], lst[o]);
  }

  return lst;
}

auto LoadLabel(const std::string& file) -> std::vector<char> {
  std::ifstream infile;
  infile.open(file, std::ios::in | std::ios::binary);

  int head[2];
  infile.read(reinterpret_cast<char*>(head), sizeof(head));

  int nlabel = b2l(head[1]);
  std::vector<char> labels(nlabel, 0);
  infile.read(labels.data(), labels.size());

  infile.close();

  return labels;
}

auto Label2Target(const std::vector<char>& labels) -> std::vector<arma::vec> {
  std::vector<arma::vec> targets(labels.size());
  for (size_t il = 0; il < labels.size(); ++il) {
    const auto& label = labels[il];
    auto& target = targets[il];
    target.resize(10);
    target.fill(0.1);
    target[label] = 0.9;
  }
  return targets;
}

auto LoadImage(const std::string& file) -> std::tuple<std::vector<std::vector<unsigned char>>, int, int> {
  std::ifstream infile;
  infile.open(file, std::ios::in | std::ios::binary);

  int head[4];
  infile.read(reinterpret_cast<char*>(head), sizeof(head));

  int nimage = b2l(head[1]);
  int width = b2l(head[2]);
  int height = b2l(head[3]);
  std::vector<std::vector<unsigned char>> images(nimage);
  for (int i = 0; i < nimage; ++i) {
    images[i].resize(width * height);
    infile.read(reinterpret_cast<char*>(images[i].data()), images[i].size());
  }

  infile.close();
  return std::make_tuple(images, width, height);
}

auto Pixel2Input(const std::vector<std::vector<unsigned char>>& pixels) -> std::vector<arma::vec> {
  std::vector<arma::vec> inputs(pixels.size());
  for (size_t ip = 0; ip < pixels.size(); ++ip) {
    const auto& pixel = pixels[ip];
    auto& input = inputs[ip];
    input.resize(pixel.size());
    for (size_t i = 0; i < pixel.size(); ++i) {
      input[i] = (double)pixel[i] / std::numeric_limits<unsigned char>::max() * 0.9 + 0.1;
    }
  }
  return inputs;
}

void PrintPixel(std::vector<unsigned char> pixel, int width) {
  for (size_t ip = 0; ip < pixel.size(); ++ip) {
    std::cout << ((pixel[ip] > std::numeric_limits<unsigned char>::max() / 2) ? "#" : " ");
    if ((ip + 1) % width == 0)
      std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    printf("iteration=? hidden=? rate1=? rate2=?\n");
    exit(1);
  }

  const int niteration = str2num<int>(argv[1]);
  const int nhidden = str2num<int>(argv[2]);
  const double rate1 = str2num<double>(argv[3]);
  const double rate2 = str2num<double>(argv[4]);

  printf("iteration=%d hidden=%d rate1=%.2lf rate2=%.2lf\n",
         niteration, nhidden, rate1, rate2);

  bpnn n;

  {
    const auto& labels = LoadLabel("data/train-labels-idx1-ubyte");
    const auto& targets = Label2Target(labels);

    const auto& images = LoadImage("data/train-images-idx3-ubyte");
    const auto& inputs = Pixel2Input(std::get<0>(images));
    auto width = std::get<1>(images);
    auto height = std::get<2>(images);

    const int nsample = labels.size();

    n.init(width * height, nhidden, 10, rate1, rate2);
    for (int i = 0; i < niteration; ++i) {
      int ith = 0;
      int correct = 0;
      for (auto is : rand_seq(nsample)) {
        int label = labels[is];
        const auto& target = targets[is];
        const auto& input = inputs[is];

        const auto& output = n.forward(input);
        n.backward(target);

        if (argmax(output) == label)
          ++correct;

        if (++ith % 1000 == 0) {
          printf("iteration: %2d, %3d%%  correct: %3.2lf%%\r",
                 i + 1,
                 (int)(ith * 100 / nsample),
                 (double)correct / ith * 100.f);
          fflush(stdout);
        }
      }
      printf("\n");
    }
  }

  {
    const auto& labels = LoadLabel("data/t10k-labels-idx1-ubyte");

    const auto& images = LoadImage("data/t10k-images-idx3-ubyte");
    const auto& pixels = std::get<0>(images);
    const auto& inputs = Pixel2Input(pixels);
    // auto width = std::get<1>(images);
    const int nsample = labels.size();

    int ith = 0;
    int correct = 0;
    for (auto is : rand_seq(nsample)) {
      int label = labels[is];
      const auto& input = inputs[is];

      const auto& output = n.forward(input);
      if (argmax(output) == label) {
        ++correct;
      } else {
        // PrintPixel(pixels[is], width);
      }

      if (++ith % 1000 == 0) {
        printf("correct: %3.2lf%%\r", correct * 100.f / ith);
        fflush(stdout);
      }
    }
    printf("\n");
  }

  return 0;
}
