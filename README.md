# -_neural_network
C++ neural network implementation with customizable architecture. Train and evaluate on diverse datasets. Save/load weights, ensuring flexibility. Ideal for machine learning experiments. Contributions welcome. MIT License.

Welcome to Galvatrone, a C++ neural network implementation with customizable architecture.

## Description

This project provides a flexible neural network framework designed for machine learning experiments and practical applications. The architecture allows customization of input sizes, hidden layer configurations, and output sizes. Train the network using backpropagation, save/load weights, and adapt it to various problem domains.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Galvatrone.git
   cd Galvatrone

Compile and Run:
Compile the C++ code and run the executable.
    g++ main.cpp -o galvatrone
./galvatrone

Training and Evaluation:
Follow the example in main.cpp 
for training the network on a dataset and evaluating its performance.

Code:

#include<iostream>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<fstream>
#include<vector>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
        : inputSize(inputSize), hiddenSize1(hiddenSize1), hiddenSize2(hiddenSize2), outputSize(outputSize) {
        weightsInputHidden1.resize(inputSize, vector<double>(hiddenSize1));
        weightsHidden1Hidden2.resize(hiddenSize1, vector<double>(hiddenSize2));
        weightsHidden2Output.resize(hiddenSize2, vector<double>(outputSize));

        initializeRandomWeights();
    }

    void initializeRandomWeights() {
        srand(static_cast<unsigned int>(time(nullptr)));

        // Initialize weights within [-1, 1] for the first layer
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize1; ++j) {
                weightsInputHidden1[i][j] = (rand() % 200) / 100.0 - 1.0;
            }
        }

        // Initialize weights within [-1, 1] for the second layer
        for (int i = 0; i < hiddenSize1; ++i) {
            for (int j = 0; j < hiddenSize2; ++j) {
                weightsHidden1Hidden2[i][j] = (rand() % 200) / 100.0 - 1.0;
            }
        }

        // Initialize weights within [-1, 1] for the output layer
        for (int i = 0; i < hiddenSize2; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weightsHidden2Output[i][j] = (rand() % 200) / 100.0 - 1.0;
            }
        }
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    vector<double> forward(const vector<double>& inputs) {
        vector<double> hiddenInputs1(hiddenSize1);
        vector<double> hiddenOutputs1(hiddenSize1);
        vector<double> hiddenInputs2(hiddenSize2);
        vector<double> hiddenOutputs2(hiddenSize2);
        vector<double> finalInputs(outputSize);
        vector<double> finalOutputs(outputSize);

        // Forward pass through the first hidden layer
        for (int i = 0; i < hiddenSize1; ++i) {
            hiddenInputs1[i] = 0;
            for (int j = 0; j < inputSize; ++j) {
                hiddenInputs1[i] += inputs[j] * weightsInputHidden1[j][i];
            }
            hiddenOutputs1[i] = sigmoid(hiddenInputs1[i]);
        }

        // Forward pass through the second hidden layer
        for (int i = 0; i < hiddenSize2; ++i) {
            hiddenInputs2[i] = 0;
            for (int j = 0; j < hiddenSize1; ++j) {
                hiddenInputs2[i] += hiddenOutputs1[j] * weightsHidden1Hidden2[j][i];
            }
            hiddenOutputs2[i] = sigmoid(hiddenInputs2[i]);
        }

        // Forward pass through the output layer
        for (int i = 0; i < outputSize; ++i) {
            finalInputs[i] = 0;
            for (int j = 0; j < hiddenSize2; ++j) {
                finalInputs[i] += hiddenOutputs2[j] * weightsHidden2Output[j][i];
            }
            finalOutputs[i] = sigmoid(finalInputs[i]);
        }

        return finalOutputs;
    }
}
void loadWeightsFromFile(const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        // Load weights for the first hidden layer
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize1; ++j) {
                file >> weightsInputHidden1[i][j];
            }
        }

        // Load weights for the second hidden layer
        for (int i = 0; i < hiddenSize1; ++i) {
            for (int j = 0; j < hiddenSize2; ++j) {
                file >> weightsHidden1Hidden2[i][j];
            }
        }

        // Load weights for the output layer
        for (int i = 0; i < hiddenSize2; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                file >> weightsHidden2Output[i][j];
            }
        }

        file.close();
    } else {
        cerr << "Failed to open weights file." << endl;
    }
}

void saveWeightsToFile(const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        // Save weights for the first hidden layer
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize1; ++j) {
                file << weightsInputHidden1[i][j] << " ";
            }
        }

        // Save weights for the second hidden layer
        for (int i = 0; i < hiddenSize1; ++i) {
            for (int j = 0; j < hiddenSize2; ++j) {
                file << weightsHidden1Hidden2[i][j] << " ";
            }
        }

        // Save weights for the output layer
        for (int i = 0; i < hiddenSize2; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                file << weightsHidden2Output[i][j] << " ";
            }
        }

        file.close();
    }
}

void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets,
           double learningRate = 0.9, int epochs = 1000000000) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (size_t example = 0; example < inputs.size(); ++example) {
            const vector<double>& input = inputs[example];
            const vector<double>& target = targets[example];

            // Forward pass
            vector<double> hiddenInputs1(hiddenSize1);
            vector<double> hiddenOutputs1(hiddenSize1);
            vector<double> hiddenInputs2(hiddenSize2);
            vector<double> hiddenOutputs2(hiddenSize2);
            vector<double> finalInputs(outputSize);
            vector<double> finalOutputs(outputSize);

            // Forward pass through the first hidden layer
            for (int i = 0; i < hiddenSize1; ++i) {
                hiddenInputs1[i] = 0;
                for (int j = 0; j < inputSize; ++j) {
                    hiddenInputs1[i] += input[j] * weightsInputHidden1[j][i];
                }
                hiddenOutputs1[i] = sigmoid(hiddenInputs1[i]);
            }

            // Forward pass through the second hidden layer
            for (int i = 0; i < hiddenSize2; ++i) {
                hiddenInputs2[i] = 0;
                for (int j = 0; j < hiddenSize1; ++j) {
                    hiddenInputs2[i] += hiddenOutputs1[j] * weightsHidden1Hidden2[j][i];
                }
                hiddenOutputs2[i] = sigmoid(hiddenInputs2[i]);
            }

            // Forward pass through the output layer
            for (int i = 0; i < outputSize; ++i) {
                finalInputs[i] = 0;
                for (int j = 0; j < hiddenSize2; ++j) {
                    finalInputs[i] += hiddenOutputs2[j] * weightsHidden2Output[j][i];
                }
                finalOutputs[i] = sigmoid(finalInputs[i]);
            }

            // Backpropagation
            vector<double> outputErrors(outputSize);
            vector<double> outputDeltas(outputSize);
            vector<double> hiddenErrors2(hiddenSize2);
            vector<double> hiddenDeltas2(hiddenSize2);
            vector<double> hiddenErrors1(hiddenSize1);
            vector<double> hiddenDeltas1(hiddenSize1);

            for (int i = 0; i < outputSize; ++i) {
                outputErrors[i] = target[i] - finalOutputs[i];
                outputDeltas[i] = outputErrors[i] * finalOutputs[i] * (1.0 - finalOutputs[i]);
            }

            for (int i = 0; i < hiddenSize2; ++i) {
                hiddenErrors2[i] = 0;
                for (int j = 0; j < outputSize; ++j) {
                    hiddenErrors2[i] += outputDeltas[j] * weightsHidden2Output[i][j];
                }
                hiddenDeltas2[i] = hiddenErrors2[i] * hiddenOutputs2[i] * (1.0 - hiddenOutputs2[i]);
            }

            for (int i = 0; i < hiddenSize1; ++i) {
                hiddenErrors1[i] = 0;
                for (int j = 0; j < hiddenSize2; ++j) {
                    hiddenErrors1[i] += hiddenDeltas2[j] * weightsHidden1Hidden2[i][j];
                }
                hiddenDeltas1[i] = hiddenErrors1[i] * hiddenOutputs1[i] * (1.0 - hiddenOutputs1[i]);
            }

            // Update weights
            for (int i = 0; i < hiddenSize1; ++i) {
                for (int j = 0; j < hiddenSize2; ++j) {
                    weightsHidden1Hidden2[i][j] += learningRate * hiddenOutputs1[i] * hiddenDeltas2[j];
                }
            }

            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < hiddenSize1; ++j) {
                    weightsInputHidden1[i][j] += learningRate * input[i] * hiddenDeltas1[j];
                }
            }

            for (int i = 0; i < hiddenSize2; ++i) {
                for (int j = 0; j < outputSize; ++j) {
                    weightsHidden2Output[i][j] += learningRate * hiddenOutputs2[i] * outputDeltas[j];
                }
            }

            // Calculate total error
            for (int i = 0; i < outputSize; ++i) {
                totalError += abs(outputErrors[i]);
            }
        }

        // Print error every 10000 epochs
        if (epoch % 10000 == 0) {
            cout << "Epoch: " << epoch << ", Error: " << totalError << endl;
            saveWeightsToFile("weights.txt");
        }

        // If error is sufficiently small, stop training
        if (totalError < 0.01) {
            cout << "Training completed. Epoch: " << epoch << ", Error: " << totalError << endl;
            break;
        }
        // Save weights after each epochint main() {
    int inputSize = 4;
    int hiddenSize1 = 6;  // New size for the first hidden layer
    int hiddenSize2 = 2;  // New size for the second hidden layer
    int outputSize = 1;

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    neuralNet.loadWeightsFromFile("weights.txt");

    // Example input data and target values
    vector<vector<double>> inputs = {{0, 0, 1, 1}, {1, 1, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1}, {0, 1, 0, 0}, {0, 1, 2, 3}, {0, 2, 2, 1}};
    vector<vector<double>> targets = {{1}, {1}, {1}, {0}, {0}, {1}, {2}, {1}, {0}, {1}};

    // Train the network
    neuralNet.train(inputs, targets);

    // Evaluate the network
    cout << "Results after training:" << endl;
    for (const auto& input : inputs) {
        vector<double> predictions = neuralNet.forward(input);
        cout << "Input: " << input[0] << ", " << input[1] << ',' << input[2] << ", " << input[3] << " Output: " << predictions[0] << endl;
    }

    // Save weights after training
    neuralNet.saveWeightsToFile("weights.txt");
    return 0;
}

        //saveWeightsToFile("weights.txt");
    }
}

private:
    int inputSize;
    int hiddenSize1;
    int hiddenSize2;
    int outputSize;

    vector<vector<double>> weightsInputHidden1;
    vector<vector<double>> weightsHidden1Hidden2;
    vector<vector<double>> weightsHidden2Output;
};
int main() {
    int inputSize = 4;
    int hiddenSize1 = 6;  // New size for the first hidden layer
    int hiddenSize2 = 2;  // New size for the second hidden layer
    int outputSize = 1;

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    neuralNet.loadWeightsFromFile("weights.txt");

    // Example input data and target values
    vector<vector<double>> inputs = {{0, 0, 1, 1}, {1, 1, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1}, {0, 1, 0, 0}, {0, 1, 2, 3}, {0, 2, 2, 1}};
    vector<vector<double>> targets = {{1}, {1}, {1}, {0}, {0}, {1}, {2}, {1}, {0}, {1}};

    // Train the network
    neuralNet.train(inputs, targets);

    // Evaluate the network
    cout << "Results after training:" << endl;
    for (const auto& input : inputs) {
        vector<double> predictions = neuralNet.forward(input);
        cout << "Input: " << input[0] << ", " << input[1] << ',' << input[2] << ", " << input[3] << " Output: " << predictions[0] << endl;
    }

    // Save weights after training
    neuralNet.saveWeightsToFile("weights.txt");
    return 0;
}

Contributing:
Contributions are welcome!
 Feel free to open issues, suggest improvements,
  or submit pull requests.
