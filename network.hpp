#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<< (std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d:     return os << "Conv2d";
        case LayerType::Linear:     return os << "Linear";
        case LayerType::MaxPool2d:  return os << "MaxPool2d";
        case LayerType::ReLu:       return os << "ReLu";
        case LayerType::SoftMax:    return os << "SoftMax";
        case LayerType::Flatten:    return os << "Flatten";
    };
    return os << static_cast<std::uint8_t>(layer_type);
}

class Layer {
    public:
        Layer(LayerType layer_type) : layer_type_(layer_type), input_(), weights_(), bias_(), output_() {}

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) = 0;

        void print() {
            std::cout << layer_type_ << std::endl;
            if (!input_.empty())   std::cout << "  input: "   << input_   << std::endl;
            if (!weights_.empty()) std::cout << "  weights: " << weights_ << std::endl;
            if (!bias_.empty())    std::cout << "  bias: "    << bias_    << std::endl;
            if (!output_.empty())  std::cout << "  output: "  << output_  << std::endl;
        }
        
        // TODO: additional required methods

     LayerType getLayerType()  {
        return layer_type_;
    }

    Tensor &getInput()  {
        return input_;
    }

    Tensor &getWeights()  {
        return weights_;
    }

    Tensor &getBias()  {
        return bias_;
    }

    Tensor &getOutput()  {
        return output_;
    }

    
    void setInput(Tensor &input) {
        input_ = input;
    }

    void setWeights(Tensor &weights) {
        weights_ = weights;
    }

    void setBias(Tensor &bias) {
        bias_ = bias;
    }

    void setOutput(Tensor &output) {
        output_ = output;
    }

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;
};


class Conv2d : public Layer {
    public:
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::Conv2d) {
        this->in_channels_=in_channels;
        this->out_channels_=out_channels;
        this->kernel_size_=kernel_size;
        this->stride_=stride;
        this->pad_=pad;
        }

        
        void fwd() override{

            size_t in_height=input_.H;
            size_t in_width=input_.W;
            size_t out_height= (in_height -kernel_size_ + (2 * pad_) )/stride_ + 1;
            size_t out_width= (in_width-kernel_size_+(2* pad_))/stride_ +1;
            output_=Tensor(input_.N,out_channels_,out_height,out_width);

            
        for (size_t n = 0; n < input_.N; ++n) {
            for (size_t out_c = 0; out_c < out_channels_; ++out_c) {
                for (size_t out_h = 0; out_h < out_height; ++out_h) {
                    for (size_t out_w = 0; out_w < out_width; ++out_w) {
                        float value = bias_(out_c);
                        for (size_t in_c = 0; in_c < in_channels_; ++in_c) {
                            for (size_t k_h = 0; k_h < kernel_size_; ++k_h) {
                                for (size_t k_w = 0; k_w < kernel_size_; ++k_w) {
                                    size_t in_h = out_h * stride_ + k_h - pad_;
                                    size_t in_w = out_w * stride_ + k_w - pad_;
                                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                        value += input_(n, in_c, in_h, in_w) * weights_(out_c, in_c, k_h, k_w);
                                    }
                                }
                            }
                        }
                        output_(n, out_c, out_h, out_w) = value;
                    }
                }
            }
        }
    
        }
    void read_weights_bias(std::ifstream& is) override{
        
        size_t weights_size= out_channels_*in_channels_*kernel_size_*kernel_size_;
        size_t bias_size=out_channels_;
        weights_ = Tensor(out_channels_, in_channels_, kernel_size_, kernel_size_);
        bias_ = Tensor(out_channels_);

    
        is.read(reinterpret_cast<char*>(weights_.data()), weights_size * sizeof(float));

        is.read(reinterpret_cast<char*>(bias_.data()), bias_size * sizeof(float));
        
        is.close();

    }
    
    
    Tensor get_output(){
        return this->output_;
    }
    public:

        size_t in_channels_;
        size_t out_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t pad_; 
        Tensor weights_;
        Tensor bias_;
};


class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear) {}
    // TODO
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d) {}
    // TODO
};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {
        }
    // TODO
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    // TODO
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    // TODO
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            // TODO
        }

        void load(std::string file) {
            // TODO
        }

        //Tensor predict(Tensor input) {
            // TODO
        //}

    private:
        bool debug_;
        // TODO: storage for layers
};

#endif 
