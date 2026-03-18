#pragma once
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>


using v_ui8 = Eigen::VectorX<uint8_t>; 

class Dataset {

public:
    Dataset(const YAML::Node& cfg, bool is_train);
    std::pair<float*, size_t> request_imgs(const size_t n, v_ui8& label_batch);
    std::pair<float*, size_t> request_val_imgs(const size_t n);
    constexpr void set_val() noexcept {at = cut_off_element;};

private:
    std::vector<float> images; /**< `std::vector` storing image data in consecutively stored blocks of `img_size` datapoints. */
    std::vector<uint8_t> labels; /**< `std::vector` of the corresponding labels to `images` . */
    
    size_t cut_off_element; /**< First element of the validation subset. */
    size_t num_pixels; /**< Number of datapoints per image.*/
    size_t size; /**< Number of images in dataset. */
    size_t at; /**< index of current image. */

    void load_binary(const std::string& image_path,
                     const std::string& label_path,
                     const bool shuffle);
    

};
