#pragma once

#include <armadillo>
#include <cmath>
#include <string>

namespace sml {

class bpnn {
 public:
  bpnn() = default;

  bool dump(const std::string& file) const;
  bool load(const std::string& file);
  void init(int ninput, int nhidden, int noutput, double rate1, double rate2);

  const arma::vec& forward(const arma::vec& input);
  void backward(const arma::vec& target);

 private:
  arma::vec input_;
  arma::vec hidden_;
  arma::vec output_;
  arma::mat hiddenweight_;
  arma::vec hiddenbias_;
  arma::mat outputweight_;
  arma::vec outputbias_;

  arma::mat hiddenwchanged_;
  arma::vec hiddenbchanged_;
  arma::mat outputwchanged_;
  arma::vec outputbchanged_;

  double rate1_;
  double rate2_;
};

}  // namespace sml
