#include "bpnn.h"
#include <cmath>
#include <iostream>
using namespace sml;

static inline double sigmod(double v) {
  return 1.f / (1.f + std::exp(-v));
}

static inline double dsigmod(double v) {
  return v * (1.f - v);
}

bool bpnn::dump(const std::string& file) const {
  return true;
}

bool bpnn::load(const std::string& file) {
  return true;
}

void bpnn::init(int ninput, int nhidden, int noutput, double rate1, double rate2) {
  rate1_ = rate1;
  rate2_ = rate2;

  hiddenweight_.zeros(nhidden, ninput);
  hiddenwchanged_.zeros(nhidden, ninput);
  hiddenbias_.zeros(nhidden);
  hiddenbchanged_.zeros(nhidden);

  outputweight_.zeros(noutput, nhidden);
  outputwchanged_.zeros(noutput, nhidden);
  outputbias_.zeros(noutput);
  outputbchanged_.zeros(noutput);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-100, 100);
  auto rand = [&](double) { return dis(gen) / 100.1f; };
  hiddenweight_.transform(rand);
  hiddenbias_.transform(rand);
  outputweight_.transform(rand);
  outputbias_.transform(rand);
}

const arma::vec& bpnn::forward(const arma::vec& input) {
  input_ = arma::vec(input);

  hidden_ = hiddenweight_ * input_ + hiddenbias_;
  hidden_.transform(sigmod);

  output_ = outputweight_ * hidden_ + outputbias_;
  output_.transform(sigmod);

  return output_;
}

void bpnn::backward(const arma::vec& target) {
  const auto& err_output = target - output_;
  auto doutput(output_);
  doutput.transform(dsigmod);
  const auto& grad_output = err_output % doutput;

  const auto& err_hidden = outputweight_.t() * grad_output;
  auto dhidden(hidden_);
  dhidden.transform(dsigmod);
  const auto& grad_hidden = err_hidden % dhidden;

  hiddenwchanged_ = grad_hidden * input_.t() * rate1_ + hiddenwchanged_ * rate2_;
  hiddenweight_ += hiddenwchanged_;
  hiddenbchanged_ = grad_hidden * rate1_ + hiddenbchanged_ * rate2_;
  hiddenbias_ += hiddenbchanged_;

  outputwchanged_ = grad_output * hidden_.t() * rate1_ + outputwchanged_ * rate2_;
  outputweight_ += outputwchanged_;
  outputbchanged_ = grad_output * rate1_ + outputbchanged_ * rate2_;
  outputbias_ += outputbchanged_;
}
