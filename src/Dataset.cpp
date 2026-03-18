#include "Dataset.h"

#include <stdexcept>
#include <cassert>

#include <fstream>
#include <random>


uint32_t read_be_uint32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

std::pair<float*, size_t> Dataset::request_imgs(const size_t n, v_ui8& label_batch) {
    if (at + n > cut_off_element) return {nullptr, num_pixels};
    label_batch = Eigen::Map<v_ui8>(&labels[at - 1], n);
    at += n;
    return {images.data() + (at - 1 - n) * num_pixels, num_pixels};
};

std::pair<float*, size_t> Dataset::request_val_imgs(const size_t n) {
    if (at + n <= cut_off_element || at + n > size) return {nullptr, num_pixels};
    at += n;
    return {images.data() + at - n - 1, num_pixels};
};

Dataset::Dataset(const YAML::Node& cfg, bool is_train)
{
    float val_ratio = cfg["val_ratio"].as<float>();
    if (val_ratio >= 1 && val_ratio < 0) throw std::domain_error("Domain not satisfied: 0 <= val_ratio < 1");

    num_pixels = cfg["image_size"]["height"].as<int>() * cfg["image_size"]["width"].as<int>();
    std::string dataset_dir = cfg["dataset_dir"].as<std::string>();

    std::string img_path = is_train ?
        dataset_dir + "/" + cfg["train_images"].as<std::string>() :
        dataset_dir + "/" + cfg["test_images"].as<std::string>();

    std::string lbl_path = is_train ?
        dataset_dir + "/" + cfg["train_labels"].as<std::string>() :
        dataset_dir + "/" + cfg["test_labels"].as<std::string>();

    load_binary(img_path, lbl_path, cfg["shuffle"].as<bool>());
    cut_off_element = size - static_cast<size_t>(size * val_ratio);
}

void Dataset::load_binary(const std::string& image_path,
                          const std::string& label_path, const bool shuffle)
{
    // initialize filestreams 
    std::ifstream img(image_path, std::ios::binary);
    std::ifstream lbl(label_path, std::ios::binary);

    // enforce file existence
    if (!img.is_open()) throw std::runtime_error("Cannot open: " + image_path);
    if (!lbl.is_open()) throw std::runtime_error("Cannot open: " + label_path);

    // MNIST format headers
    read_be_uint32(img); // disregard first 4 bytes (contains magic number)
    read_be_uint32(lbl); // disregard first 4 bytes (contains magic number)
    
    // fetch dataset dimensions 
    uint32_t num_images = read_be_uint32(img);
    uint32_t num_labels = read_be_uint32(lbl);


    // next 2 reads from image file contain image dimensions -> compare to expected number of pixels
    if (read_be_uint32(img) * read_be_uint32(img) != num_pixels) 
        throw std::runtime_error("num_pixels in config does not match MNIST file");
    
    // check that dataset size matches number of labels
    if (num_images != num_labels)
        throw std::runtime_error("Image/label count mismatch");


    // label buffer 
    uint8_t* lbl_buf = (uint8_t *)malloc(num_labels); 
    
    // read labels
    lbl.read((char*)lbl_buf, num_labels);
    lbl.close();
    
    // temporary buffer for image data
    // less overhead than std::vetor -> faster
    uint8_t* img_buf = (uint8_t *)malloc(num_labels * num_pixels); 
    
    // read images
    img.read((char*)img_buf, num_labels * num_pixels);
    img.close();
    

    // preallocate and initialize label vector
    labels.resize(num_images);
    // preallocate and initialize image vector 
    images.resize(num_images * num_pixels); // very expensive
    

    std::vector<size_t> idx(num_labels);
    std::iota(idx.begin(), idx.end(),0);
    
    if (shuffle) [[likely]] {
        std::random_device dev;
        std::mt19937 gen(dev());
        std::shuffle(idx.begin(), idx.end(), gen);
    }
    // populate data structures for label and images in the shuffeled order

    
    for (size_t i{0}; i < num_labels; i++){
        labels[i] = lbl_buf[idx[i]];
        
        // format and normalize shuffeled input to floating point numbers [0,1] representing greyscale values
        std::transform(img_buf + num_pixels * idx[i]
                        , img_buf + num_pixels * (idx[i]+1)
                        , images.begin() + num_pixels * i
                        , [](const uint8_t a){return static_cast<float>(a) / 255;});
    }

    // avoid memory leak
    free(lbl_buf);
    free(img_buf);

    // set recource distribution counters
    size = num_labels;
    at = 1;
    
    return;
}

