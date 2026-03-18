#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <yaml-cpp/yaml.h>


class Dataset {
public:
    Dataset(const YAML::Node& cfg, bool is_train);

    const std::vector<float>& images() const { return images_; };
    const std::vector<uint8_t>& labels() const { return labels_; };

    size_t size() const { return num_samples_; }
    size_t pixels() const { return num_pixels_; }

    float* image_ptr(size_t i) {
        return images_.data() + i * num_pixels_;
    }
    

    uint8_t label(size_t i) const {
        return labels_[i];
    }
    
    uint8_t& label(size_t i) {
        return labels_[i];
    }

private:
    size_t num_samples_{0};
    size_t num_pixels_{0};
    bool is_train_;

    std::vector<float> images_;   // [N * pixels]
    std::vector<uint8_t> labels_; // [N]

    void load_binary(const std::string& image_path,
                     const std::string& label_path, const bool shuffle);
};

struct DatasetView {
    Dataset* dataset;
    size_t begin;
    size_t end;

    size_t size() const { return end - begin; }
};
