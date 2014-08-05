#ifndef __BPNN_H__
#define __BPNN_H__

#include <string>
#include <cmath>
#include <armadillo>

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
    arma::vec _input;
    arma::vec _hidden;
    arma::vec _output;
    arma::mat _hiddenweight;
    arma::vec _hiddenbias;
    arma::mat _outputweight;
    arma::vec _outputbias;

    arma::mat _hiddenwchanged;
    arma::vec _hiddenbchanged;
    arma::mat _outputwchanged;
    arma::vec _outputbchanged;

    double _rate1;
    double _rate2;
};

}

#endif
