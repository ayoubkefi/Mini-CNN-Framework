
#include "mnist.hpp"
#include "network.hpp"
#include "iostream"
#include "fstream"
#include "string"
int main() {

    std::string test_dataset_path = "t10k-images-idx3-ubyte";
   
    
    NeuralNetwork LeNet;
    std::ifstream is("lenet.raw", std::ios::binary);
    //adding the layer 
    LeNet.add(new Conv2d(1,6,5,1,2));
    
    LeNet.add(new ReLu);
    LeNet.add(new MaxPool2d(2,2));
     LeNet.add(new Conv2d(6,16,5));
    LeNet.add(new ReLu);
    LeNet.load("lenet.raw");
    LeNet.add(new MaxPool2d(2,2));
    LeNet.add(new Flatten);
   
   
   
    MNIST test_mnist1(test_dataset_path);
    Tensor input1=test_mnist1.at(1);
    test_mnist1.print(1);
    Tensor output=LeNet.predict(input1);
    LeNet.layers_[0]->print();
    LeNet.layers_[3]->print();
    output.display();
    
    

    


   
    
    return 0;
}
