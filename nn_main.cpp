#include <iostream>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <string>


using namespace std;

const string WEIGHTS_FILE = "letters_weights.txt";    // weights trained by NN_Training
const string SCALING_FILE = "letters_scaling.txt";    // feature min and max from train

const int FEATURE_COUNT = 16;     // features per letter
const int CLASS_COUNT = 26;       // letters A..Z

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
        : inputSize(inputSize), hiddenSize1(hiddenSize1), hiddenSize2(hiddenSize2), outputSize(outputSize),
          weightsInputHidden1(nullptr), weightsHidden1Hidden2(nullptr), weightsHidden2Output(nullptr),
          biasHidden1(nullptr), biasHidden2(nullptr), biasOutput(nullptr) {
        if (inputSize <= 0 || hiddenSize1 <= 0 || hiddenSize2 <= 0 || outputSize <= 0) {
            throw invalid_argument("Layer sizes must be positive.");
        }
        try {
            weightsInputHidden1 = allocateMatrix(inputSize, hiddenSize1);
            weightsHidden1Hidden2 = allocateMatrix(hiddenSize1, hiddenSize2);
            weightsHidden2Output = allocateMatrix(hiddenSize2, outputSize);
            biasHidden1 = new double[hiddenSize1]();
            biasHidden2 = new double[hiddenSize2]();
            biasOutput = new double[outputSize]();
        } catch (...) {
            deleteMatrix(weightsInputHidden1, inputSize);
            deleteMatrix(weightsHidden1Hidden2, hiddenSize1);
            deleteMatrix(weightsHidden2Output, hiddenSize2);
            delete[] biasHidden1;
            delete[] biasHidden2;
            delete[] biasOutput;
            throw;
        }
    }

    ~NeuralNetwork() {
        deleteMatrix(weightsInputHidden1, inputSize);
        deleteMatrix(weightsHidden1Hidden2, hiddenSize1);
        deleteMatrix(weightsHidden2Output, hiddenSize2);
        delete[] biasHidden1;
        delete[] biasHidden2;
        delete[] biasOutput;
    }

    // The network owns its arrays: shallow copying is not allowed.
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    void forward(const double* inputs, double* finalOutputs) {
        double* buffer = new double[2 * static_cast<size_t>(hiddenSize1) + 2 * static_cast<size_t>(hiddenSize2) + outputSize];
        double* hiddenInputs1 = buffer;
        double* hiddenOutputs1 = hiddenInputs1 + hiddenSize1;
        double* hiddenInputs2 = hiddenOutputs1 + hiddenSize1;
        double* hiddenOutputs2 = hiddenInputs2 + hiddenSize2;
        double* finalInputs = hiddenOutputs2 + hiddenSize2;

        // Forward pass through the first hidden layer
        for (int i = 0; i < hiddenSize1; ++i) {
            hiddenInputs1[i] = biasHidden1[i];
            for (int j = 0; j < inputSize; ++j) {
                hiddenInputs1[i] += inputs[j] * weightsInputHidden1[j][i];
            }
            hiddenOutputs1[i] = sigmoid(hiddenInputs1[i]);
        }

        // Forward pass through the second hidden layer
        for (int i = 0; i < hiddenSize2; ++i) {
            hiddenInputs2[i] = biasHidden2[i];
            for (int j = 0; j < hiddenSize1; ++j) {
                hiddenInputs2[i] += hiddenOutputs1[j] * weightsHidden1Hidden2[j][i];
            }
            hiddenOutputs2[i] = sigmoid(hiddenInputs2[i]);
        }

        // Forward pass through the output layer
        for (int i = 0; i < outputSize; ++i) {
            finalInputs[i] = biasOutput[i];
            for (int j = 0; j < hiddenSize2; ++j) {
                finalInputs[i] += hiddenOutputs2[j] * weightsHidden2Output[j][i];
            }
            finalOutputs[i] = sigmoid(finalInputs[i]);
        }

        delete[] buffer;
    }

    // Reads the weights into a temporary array and copies them into the network only
    // after the whole file is checked: on any error the current weights stay unchanged.
    bool loadWeightsFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Cannot open weights file: " << filename << endl;
            return false;
        }

        const size_t count = weightCount();
        double* values = new double[count];
        for (size_t k = 0; k < count; ++k) {
            if (!(file >> values[k]) || !isfinite(values[k])) {
                cerr << "Weights file is damaged or has fewer than " << count << " numbers: " << filename << endl;
                delete[] values;
                return false;
            }
        }
        double extra;
        if (file >> extra || !file.eof()) {
            cerr << "Weights file has more than " << count << " numbers or extra data"
                 << " (different network architecture?): " << filename << endl;
            delete[] values;
            return false;
        }

        size_t k = 0;
        // Order: three weight matrices, then three bias arrays.
        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize1; ++j)
                weightsInputHidden1[i][j] = values[k++];
        for (int i = 0; i < hiddenSize1; ++i)
            for (int j = 0; j < hiddenSize2; ++j)
                weightsHidden1Hidden2[i][j] = values[k++];
        for (int i = 0; i < hiddenSize2; ++i)
            for (int j = 0; j < outputSize; ++j)
                weightsHidden2Output[i][j] = values[k++];
        for (int i = 0; i < hiddenSize1; ++i) biasHidden1[i] = values[k++];
        for (int i = 0; i < hiddenSize2; ++i) biasHidden2[i] = values[k++];
        for (int i = 0; i < outputSize; ++i) biasOutput[i] = values[k++];

        delete[] values;
        return true;
    }

    // Number of values in the weights file for this architecture (weights + biases).
    size_t weightCount() const {
        return static_cast<size_t>(inputSize) * hiddenSize1
             + static_cast<size_t>(hiddenSize1) * hiddenSize2
             + static_cast<size_t>(hiddenSize2) * outputSize
             + hiddenSize1 + hiddenSize2 + outputSize;
    }

private:

    static double** allocateMatrix(int rows, int columns) {
        double** matrix = new double*[rows]();
        try {
            for (int i = 0; i < rows; ++i) {
                matrix[i] = new double[columns]();
            }
        } catch (...) {
            deleteMatrix(matrix, rows);
            throw;
        }
        return matrix;
    }

    static void deleteMatrix(double** matrix, int rows) {
        if (matrix != nullptr) {
            for (int i = 0; i < rows; ++i) {
                delete[] matrix[i];
            }
            delete[] matrix;
        }
    }

     int inputSize;
    int hiddenSize1;
    int hiddenSize2;
    int outputSize;

    double** weightsInputHidden1;
    double** weightsHidden1Hidden2;
    double** weightsHidden2Output;

    double* biasHidden1;
    double* biasHidden2;
    double* biasOutput;
};


// Scaling as in training: min (first line) and max (second line)
// of each feature, found by NN_Training on train.
bool loadScaling(double* minValues, double* maxValues, const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open scaling file: " << filename << endl;
        return false;
    }
    for (int j = 0; j < FEATURE_COUNT; ++j) {
        if (!(file >> minValues[j])) {
            cerr << "Scaling file is damaged: " << filename << endl;
            return false;
        }
    }
    for (int j = 0; j < FEATURE_COUNT; ++j) {
        if (!(file >> maxValues[j])) {
            cerr << "Scaling file is damaged: " << filename << endl;
            return false;
        }
    }
    return true;
}


int main() {
    int inputSize = FEATURE_COUNT;   // 16 features
    int hiddenSize1 = 32;            // 2 neurons for each input feature
    int hiddenSize2 = 52;            // 2 neurons for each letter
    int outputSize = CLASS_COUNT;    // 26 letters

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    // Predictions make no sense without valid weights.
    if (!neuralNet.loadWeightsFromFile(WEIGHTS_FILE)) {
        return 1;
    }
    double minValues[FEATURE_COUNT];
    double maxValues[FEATURE_COUNT];
    if (!loadScaling(minValues, maxValues, SCALING_FILE)) {
        return 1;
    }

    int N;
    cout << "Enter the number of examples: ";
    if (!(cin >> N) || N < 0) {
        cerr << "Invalid number of examples." << endl;
        return 1;
    }

    // For each example the network takes the 16 features of a letter (numbers 0..15),
    // like a line of ocrData.txt without the letter.
    double (*newInputs)[FEATURE_COUNT] = new double[N][FEATURE_COUNT]();
    for (int i = 0; i < N; ++i) {
        cout << "Enter " << FEATURE_COUNT << " features for example " << i + 1 << ": ";
        for (int j = 0; j < inputSize; ++j) {
            if (!(cin >> newInputs[i][j])) {
                cerr << "Invalid input value." << endl;
                delete[] newInputs;
                return 1;
            }
        }
    }

    double* prediction = new double[outputSize];
    double scaled[FEATURE_COUNT];
    for (int i = 0; i < N; ++i) {
        const double* newInput = newInputs[i];
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            double range = maxValues[j] - minValues[j];
            scaled[j] = range > 0 ? (newInput[j] - minValues[j]) / range : 0.0;
        }
        neuralNet.forward(scaled, prediction);

        // The letter is the output with the largest value
        int best = 0;
        for (int k = 1; k < outputSize; ++k) {
            if (prediction[k] > prediction[best]) best = k;
        }
        cout << "Input:";
        for (int j = 0; j < FEATURE_COUNT; ++j) cout << (j ? "," : " ") << newInput[j];
        cout << " Prediction: " << static_cast<char>('A' + best)
             << " (output " << prediction[best] << ")" << endl;
    }
    delete[] prediction;
    delete[] newInputs;
    return 0;
}
