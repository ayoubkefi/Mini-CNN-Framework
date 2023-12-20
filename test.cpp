
#include "mnist.hpp"
#include "network.hpp"
#include "iostream"
#include "fstream"
#include "string"
int main() {

    std::string test_dataset_path = "t10k-images-idx3-ubyte";
    std::string biaises= "lenet.raw";
    std::ifstream is("lenet.raw", std::ios::binary);
    MNIST test_mnist(test_dataset_path);
    for(int i=1;i<100;i++){
    test_mnist.print(i);
    }      
    Conv2d lenet(1,1,3);
    std::cout<< "input channel:" <<lenet.in_channels_<<std::endl;
    std::cout<<"output channel:"<< lenet.out_channels_<<std::endl;
    std::cout<<"kernel size "<< lenet.kernel_size_<<std::endl;
    std::cout<<"stride"<<lenet.stride_<<std::endl;
    std::cout<<"padding"<<lenet.pad_<<std::endl;
    
    lenet.read_weights_bias(is);
    lenet.print();
    Tensor input = test_mnist.at(0);
    std::cout<<"\ninput Tensor:"<<std::endl;
    input.display();

    std::cout << "\nWeights Tensor:" << lenet.weights_.slice(0,1)<<std::endl;
    lenet.weights_.slice(0,1).display();

     std::cout << "\nBiases Tensor:" <<lenet.bias_.slice(0,1)<<std::endl;
    lenet.bias_.slice(0,1).display();
    lenet.setInput(input);
    lenet.print();
    lenet.fwd();
    lenet.print();
    std::cout << "output Tensor:" << lenet.getOutput().slice(0,1)<<std::endl;
    lenet.get_output().display();
    
    
    
    

   
    
    return 0;
}
