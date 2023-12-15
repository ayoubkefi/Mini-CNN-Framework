
#include "mnist.hpp"
int main() {

    std::string test_dataset_path = "t10k-images-idx3-ubyte";
    
    MNIST test_mnist(test_dataset_path);
    
    for(int i=1;i<1000;i++){
    test_mnist.print(i);
    }


    
    return 0;
}
