#ifndef UTILS_RNG_H
#define UTILS_RNG_H

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

namespace utils {
class RandomNumberGenerator {
    // Mersenne Twister random number generator.
    std::mt19937 rng;

public:
    RandomNumberGenerator(); // Seed with a value depending on time and process ID.
    explicit RandomNumberGenerator(int seed);
    RandomNumberGenerator(const RandomNumberGenerator &) = delete;
    RandomNumberGenerator &operator=(const RandomNumberGenerator &) = delete;
    ~RandomNumberGenerator();

    void seed(int seed);

    // Return random double in [0..1).
    double operator()() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        return distribution(rng);
    }

    // Return random integer in [0..bound).
    int operator()(int bound) {
        assert(bound > 0);
        std::uniform_int_distribution<int> distribution(0, bound - 1);
        return distribution(rng);
    }

    template<typename T>
    typename std::vector<T>::const_iterator choose(const std::vector<T> &vec) {
        return vec.begin() + operator()(vec.size());
    }

    template<typename T>
    typename std::vector<T>::iterator choose(std::vector<T> &vec) {
        return vec.begin() + operator()(vec.size());
    }

    template<typename T>
    void shuffle(std::vector<T> &vec) {
        std::shuffle(vec.begin(), vec.end(), rng);
    }

    template<typename T>
    void sample(std::vector<T> &vec, int n, std::vector<T> &out) {
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        std::vector<int> selected_indices;
        for (int i = 0; i < n; ++i) {
            selected_indices.push_back(indices[i]);
        }
        std::sort(selected_indices.begin(), selected_indices.end());
        for (auto ind: selected_indices) {
            out.push_back(vec[ind]);
        }     
    }

};
}

#endif
