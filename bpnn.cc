#include "bpnn.h"
#include <cmath>
#include <iostream>
using namespace sml;

static inline double sigmod(double v)
{
    return 1.f / (1.f + std::exp(-v));
}

static inline double dsigmod(double v)
{
    return v * (1.f - v);
}

bool bpnn::dump(const std::string& file) const
{
    return true;
}

bool bpnn::load(const std::string& file)
{
    return true;
}

void bpnn::init(int ninput, int nhidden, int noutput, double rate1, double rate2)
{
    _rate1 = rate1;
    _rate2 = rate2;

    _hiddenweight.zeros(nhidden, ninput);
    _hiddenwchanged.zeros(nhidden, ninput);
    _hiddenbias.zeros(nhidden);
    _hiddenbchanged.zeros(nhidden);

    _outputweight.zeros(noutput, nhidden);
    _outputwchanged.zeros(noutput, nhidden);
    _outputbias.zeros(noutput);
    _outputbchanged.zeros(noutput);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<double> dis(-100, 100);
    auto rand = [&](double) { return dis(gen)/100.1f; };
    _hiddenweight.transform(rand);
    _hiddenbias.transform(rand);
    _outputweight.transform(rand);
    _outputbias.transform(rand);
}

const arma::vec& bpnn::forward(const arma::vec& input)
{
    _input = arma::vec(input);

    _hidden = _hiddenweight * _input + _hiddenbias;
    _hidden.transform(sigmod);

    _output = _outputweight * _hidden + _outputbias;
    _output.transform(sigmod);

    return _output;
}

void bpnn::backward(const arma::vec& target)
{
    const auto& err_output = target - _output;
    auto doutput(_output);
    doutput.transform(dsigmod);
    const auto& grad_output = err_output % doutput;

    const auto& err_hidden = _outputweight.t() * grad_output; 
    auto dhidden(_hidden);
    dhidden.transform(dsigmod);
    const auto& grad_hidden = err_hidden % dhidden;

    _hiddenwchanged = grad_hidden * _input.t() * _rate1 + _hiddenwchanged * _rate2;
    _hiddenweight += _hiddenwchanged;
    _hiddenbchanged = grad_hidden * _rate1 + _hiddenbchanged * _rate2;
    _hiddenbias += _hiddenbchanged;

    _outputwchanged = grad_output * _hidden.t() * _rate1 + _outputwchanged * _rate2;
    _outputweight += _outputwchanged;
    _outputbchanged = grad_output * _rate1 + _outputbchanged * _rate2;
    _outputbias += _outputbchanged;
}
