#include "Dataset.h"
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <random>


static uint32_t read_be_uint32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  |
            uint32_t(b[3]);
}

Dataset::Dataset(const YAML::Node& cfg, bool is_train)
    : is_train_(is_train)
{
    size_t h = cfg["image_size"]["height"].as<size_t>();
    size_t w = cfg["image_size"]["width"].as<size_t>();
    num_pixels_ = h * w;

    std::string dataset_dir = cfg["dataset_dir"].as<std::string>();

    std::string img_path = dataset_dir + "/" + 
        (is_train ? cfg["train_images"].as<std::string>() 
                  : cfg["test_images"].as<std::string>());

    std::string lbl_path = dataset_dir + "/" +
    (is_train ? cfg["train_labels"].as<std::string>()
              : cfg["test_labels"].as<std::string>());

    load_binary(img_path, lbl_path, cfg["shuffle"].as<bool>());
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
    if (read_be_uint32(img) * read_be_uint32(img) != num_pixels_) 
        throw std::runtime_error("num_pixels in config does not match MNIST file");
    
    // check that dataset size matches number of labels
    if (num_images != num_labels)
        throw std::runtime_error("Image/label count mismatch");

    num_samples_ = num_images;

    // label buffer 
    uint8_t* lbl_buf = (uint8_t *)malloc(num_labels); 
    
    // read labels
    lbl.read((char*)lbl_buf, num_labels);
    lbl.close();
    
    // temporary buffer for image data
    // less overhead than std::vetor -> faster
    uint8_t* img_buf = (uint8_t *)malloc(num_labels * num_pixels_); 
    
    // read images
    img.read((char*)img_buf, num_labels * num_pixels_);
    img.close();
    

    // preallocate and initialize label vector
    labels_.resize(num_images);
    // preallocate and initialize image vector 
    images_.resize(num_images * num_pixels_); // very expensive
    

    std::vector<size_t> idx(num_labels);
    std::iota(idx.begin(), idx.end(),0);
    
    if (shuffle) [[likely]] {
        std::random_device dev;
        std::mt19937 gen(dev());
        std::shuffle(idx.begin(), idx.end(), gen);
    }

    // populate data structures for label and images in the shuffeled order
    for (size_t i{0}; i < num_labels; i++){
        labels_[i] = lbl_buf[idx[i]];
        
        // format and normalize shuffeled input to floating point numbers [0,1] representing greyscale values
        std::transform(img_buf + num_pixels_ * idx[i]
                        , img_buf + num_pixels_ * (idx[i]+1)
                        , images_.begin() + num_pixels_ * i
                        , [](const uint8_t a){return static_cast<float>(a) / 255;});
    }

    // avoid memory leak
    free(lbl_buf);
    free(img_buf);
    
    return;
}
