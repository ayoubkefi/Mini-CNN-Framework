#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>

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

    public:
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
            result=Tensor(input_.N,out_channels_,out_height,out_width);

            
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
                        result(n, out_c, out_h, out_w) = value;
                    }
                }
            }
        }
        this->output_=result;
        }
    void read_weights_bias(std::ifstream& is) override{
        
        size_t weights_size= out_channels_*in_channels_*kernel_size_*kernel_size_;
        size_t bias_size=out_channels_;
        weights_ = Tensor(out_channels_, in_channels_, kernel_size_, kernel_size_);
        bias_ = Tensor(out_channels_);

        
        is.read(reinterpret_cast<char*>(weights_.data()), weights_size * sizeof(float));
        
        is.read(reinterpret_cast<char*>(bias_.data()), bias_size * sizeof(float));
       
    this->setBias(bias_);
    this->setWeights(weights_);
        

    }
    
    
    public:

        size_t in_channels_;
        size_t out_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t pad_; 
        Tensor weights_;
        Tensor bias_;
        Tensor result;
};


class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear) {
            this->out_features_=out_features;
            this->in_features_=in_features;
        }
        
        void fwd() override{
            output_=Tensor(input_.N,out_features_);
            for (size_t n = 0; n < input_.N; ++n) {
                for (size_t out_f = 0; out_f < out_features_; ++out_f) {
                    float value = bias_(out_f);
                    for (size_t in_c = 0; in_c < input_.C; ++in_c) {
                        for (size_t in_h = 0; in_h < input_.H; ++in_h) {
                            for (size_t in_w = 0; in_w < input_.W; ++in_w) {
                                 value += input_(n, in_c, in_h, in_w) *  weights_(out_f, in_c, in_h, in_w);
                    }
                }
            }
            output_(n, out_f) = value;
            this->setOutput(output_);
        }
        }
        }
        void read_weights_bias(std::ifstream& is) override{
        
        size_t weights_size= in_features_*out_features_;
        size_t bias_size=out_features_;
        weights_ = Tensor(weights_size);
        bias_ = Tensor(bias_size);

        is.read(reinterpret_cast<char*>(weights_.data()), weights_size * sizeof(float));

        is.read(reinterpret_cast<char*>(bias_.data()), bias_size * sizeof(float));
        
        this->setBias(bias_);
        this->setWeights(weights_);

    }

    public:
        size_t out_features_;
        size_t in_features_;
        Tensor output_;

};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d) {
            this->kernel_size_=kernel_size;
            this->stride_=stride;
            this->pad_=pad;
        }
        void fwd() override {
            size_t in_height =input_.H;
            size_t in_width = input_.W;
            size_t out_height = (in_height - kernel_size_ + (2 * pad_)) / stride_ + 1;
            size_t out_width = (in_width - kernel_size_ + (2 * pad_)) / stride_ + 1;
            output_ = Tensor(input_.N, input_.C, out_height, out_width);
            for (size_t n = 0; n < input_.N; ++n) {
            for (size_t c = 0; c < input_.C; ++c) {
                for (size_t out_h = 0; out_h < out_height; ++out_h) {
                    for (size_t out_w = 0; out_w < out_width; ++out_w) {
                        float max_value = input_(n, c, out_h * stride_, out_w * stride_);
                        for (size_t k_h = 0; k_h < kernel_size_; ++k_h) {
                            for (size_t k_w = 0; k_w < kernel_size_; ++k_w) {
                                size_t in_h = out_h * stride_ + k_h - pad_;
                                size_t in_w = out_w * stride_ + k_w - pad_;
                                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                    float value = input_(n, c, in_h, in_w);
                                    max_value = (value > max_value) ? value : max_value;
                                }
                            }
                        }
                        output_(n, c, out_h, out_w) = max_value;
                        this->setOutput(output_);
                    }
                }
            }
        }
        
        }
     void read_weights_bias(std::ifstream& is) override{
            // in this layer we dont have biaised and weights !
       
            }
        
    public:
    size_t kernel_size_;
    size_t stride_;
    size_t pad_;
    Tensor output_;
};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {
        }
    void fwd() override {
        output_ = Tensor(input_.N, input_.C, input_.H, input_.W);

        for (size_t n = 0; n < input_.N; ++n) {
            for (size_t c = 0; c < input_.C; ++c) {
                for (size_t h = 0; h < input_.H; ++h) {
                    for (size_t w = 0; w < input_.W; ++w) {
                        float value = input_(n, c, h, w);
                        output_(n, c, h, w) = (value > 0) ? value : 0;
                        
                    }
                }
            }
        }
        this->setOutput(output_);
    }

    void read_weights_bias(std::ifstream& is) override {
        // in thiss  layer we dont have weights or biad
    }
public:
    Tensor output_;
};



class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
     void fwd() override {
        output_ = Tensor(input_.N, input_.C, input_.H, input_.W);
        // here we will calculate the exp of our elements and the sum of these exp
        for (size_t n = 0; n < input_.N; ++n) {
            for (size_t c = 0; c < input_.C; ++c) {
                float sum_exp = 0.0f;
                for (size_t h = 0; h < input_.H; ++h) {
                    for (size_t w = 0; w < input_.W; ++w) {
                        float exp_val = std::exp(input_(n, c, h, w));
                        output_(n, c, h, w) = exp_val;
                        sum_exp += exp_val;

                    }
                }
                
                // Now we divide by the sum of exp
                for (size_t h = 0; h < input_.H; ++h) {
                    for (size_t w = 0; w < input_.W; ++w) {
                        output_(n, c, h, w) /= sum_exp;
                        
                    }
                }
            }
            
        }
        this->setOutput(output_);   
    }

    void read_weights_bias(std::ifstream& is) override {
        // same here it's activation so no weights and biais !
    }
public:
    Tensor output_;
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
        void fwd() override {
        size_t flattened_size = input_.N * input_.C * input_.H * input_.W;
        output_ = Tensor(1, 1, 1, flattened_size);

        size_t index = 0;
        for (size_t n = 0; n < input_.N; ++n) {
            for (size_t c = 0; c < input_.C; ++c) {
                for (size_t h = 0; h < input_.H; ++h) {
                    for (size_t w = 0; w < input_.W; ++w) {
                        output_(0, 0, 0, index++) = input_(n, c, h, w);
                        this->setOutput(output_);
                    }
                }
            }
        }
    }

    void read_weights_bias(std::ifstream& is) override {
        // same here 
    }
    public:
    Tensor output_;
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            layers_.push_back(layer);
        }

        void load(std::string file) {
            std::ifstream is(file, std::ios::binary);

            for (Layer* layer : layers_) {
            layer->read_weights_bias(is);
        }
        }

        Tensor predict(Tensor input) {
            Tensor output=input;
            for (Layer* layer : layers_) {
            layer->setInput(output);
            layer->fwd();
            output = layer->output_;

        }
        return output;
        }
    private:
        bool debug_;
        
    public:
        std::vector<Layer*> layers_;
};

#endif 
