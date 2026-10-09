#include <cassert>
#include <cmath>
#include <fstream>
#include <type_traits>

#define main application_main
#include NETWORK_SOURCE
#undef main

int main() {
    // Raw owning arrays must not be shallow-copied.
    static_assert(!std::is_copy_constructible<NeuralNetwork>::value, "Owning network must not be copied");
    static_assert(!std::is_copy_assignable<NeuralNetwork>::value, "Owning network must not be assigned");
    try {
        NeuralNetwork invalid(0, 2, 3, 2);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    {
        std::ofstream weights("fixture.txt");
        for (int i = 0; i < 18; ++i) weights << "0.2 ";
    }
    double input[] = {1.0, 0.0, 1.0};
    double output[2];
    for (int repeat = 0; repeat < 100; ++repeat) {
        NeuralNetwork network(3, 2, 3, 2);
        network.loadWeightsFromFile("fixture.txt");
        network.forward(input, output);
        double h1 = 1.0 / (1.0 + std::exp(-0.4));
        double h2 = 1.0 / (1.0 + std::exp(-0.4 * h1));
        double expected = 1.0 / (1.0 + std::exp(-0.6 * h2));
        assert(std::abs(output[0] - expected) < 1e-12);
        assert(std::abs(output[1] - expected) < 1e-12);
#ifdef HAS_TRAIN
        double target[] = {1.0, 1.0};
        const double* inputs[] = {input};
        const double* targets[] = {target};
        network.train(inputs, targets, 1, 0.9, 5);
        network.forward(input, output);
        assert(output[0] > expected);
        // Captured from the original implementation with these same weights.
        assert(std::abs(output[0] - 0.67403902857262599) < 1e-12);
        assert(std::abs(output[0] - output[1]) < 1e-12);
        network.saveWeightsToFile("roundtrip.txt");
        NeuralNetwork restored(3, 2, 3, 2);
        restored.loadWeightsFromFile("roundtrip.txt");
        double restoredOutput[2];
        restored.forward(input, restoredOutput);
        assert(std::abs(restoredOutput[0] - output[0]) < 1e-6);
        std::ifstream saved("roundtrip.txt");
        double weight;
        int count = 0;
        while (saved >> weight) ++count;
        assert(count == 18);
        const double* exactTargets[] = {output};
        network.train(inputs, exactTargets, 1, 0.9, 5);
#endif
    }
}
