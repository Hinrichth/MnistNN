#include <iostream>
#include<iomanip>
void print_img(float* ptr, size_t lbl){
    std::cout << "Image with label: " << lbl << std::hex << std::setfill('.') << std::endl;
    for (size_t i{0}; i < 28; i+=2){
        for (size_t j{0}; j < 28; j++){
            float tmp = ptr[j * 28 + i];
            std::cout << ((tmp > 0.3) ? '#' : ((tmp > 0.1) ? '+' : ' '));
        }
        std::cout << std::endl;
    }
    std::cout << std::dec << std::endl;
}
void print_img(Eigen::Block<Eigen::MatrixXf, -1, 1, true> v){
    for (size_t i{0}; i < 28; i+=2){
        for (size_t j{0}; j < 28; j++){
            float tmp = v(j * 28 + i);
            std::cout << ((tmp > 0.3) ? '#' : ((tmp > 0.1) ? '+' : ' '));
        }
        std::cout << std::endl;
    }
    std::cout << std::dec << std::endl;
}