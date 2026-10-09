#include <iostream>
#include <cmath>
//#include <cstdlib>
//#include <ctime>
#include <fstream>
#include <stdexcept>
#include <clocale>


using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
        : inputSize(inputSize), hiddenSize1(hiddenSize1), hiddenSize2(hiddenSize2), outputSize(outputSize),
          weightsInputHidden1(nullptr), weightsHidden1Hidden2(nullptr), weightsHidden2Output(nullptr) {
        if (inputSize <= 0 || hiddenSize1 <= 0 || hiddenSize2 <= 0 || outputSize <= 0) {
            throw invalid_argument("Размеры слоёв должны быть положительными.");
        }
        try {
            weightsInputHidden1 = allocateMatrix(inputSize, hiddenSize1);
            weightsHidden1Hidden2 = allocateMatrix(hiddenSize1, hiddenSize2);
            weightsHidden2Output = allocateMatrix(hiddenSize2, outputSize);
        } catch (...) {
            deleteMatrix(weightsInputHidden1, inputSize);
            deleteMatrix(weightsHidden1Hidden2, hiddenSize1);
            deleteMatrix(weightsHidden2Output, hiddenSize2);
            throw;
        }
    }

    ~NeuralNetwork() {
        deleteMatrix(weightsInputHidden1, inputSize);
        deleteMatrix(weightsHidden1Hidden2, hiddenSize1);
        deleteMatrix(weightsHidden2Output, hiddenSize2);
    }

    // Массивы принадлежат сети: поверхностное копирование запрещено.
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

        // Прямой проход через первый скрытый слой
        for (int i = 0; i < hiddenSize1; ++i) {
            hiddenInputs1[i] = 0;
            for (int j = 0; j < inputSize; ++j) {
                hiddenInputs1[i] += inputs[j] * weightsInputHidden1[j][i];
            }
            hiddenOutputs1[i] = sigmoid(hiddenInputs1[i]);
        }

        // Прямой проход через второй скрытый слой
        for (int i = 0; i < hiddenSize2; ++i) {
            hiddenInputs2[i] = 0;
            for (int j = 0; j < hiddenSize1; ++j) {
                hiddenInputs2[i] += hiddenOutputs1[j] * weightsHidden1Hidden2[j][i];
            }
            hiddenOutputs2[i] = sigmoid(hiddenInputs2[i]);
        }

        // Прямой проход через выходной слой
        for (int i = 0; i < outputSize; ++i) {
            finalInputs[i] = 0;
            for (int j = 0; j < hiddenSize2; ++j) {
                finalInputs[i] += hiddenOutputs2[j] * weightsHidden2Output[j][i];
            }
            finalOutputs[i] = sigmoid(finalInputs[i]);
        }

        delete[] buffer;
    }
    void loadWeightsFromFile(const string& filename) {
        ifstream file(filename);
        if (file.is_open()) {
            // Загрузка весов для первого скрытого слоя
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < hiddenSize1; ++j) {
                    file >> weightsInputHidden1[i][j];
                }
            }

            // Загрузка весов для второго скрытого слоя
            for (int i = 0; i < hiddenSize1; ++i) {
                for (int j = 0; j < hiddenSize2; ++j) {
                    file >> weightsHidden1Hidden2[i][j];
                }
            }

            // Загрузка весов для выходного слоя
            for (int i = 0; i < hiddenSize2; ++i) {
                for (int j = 0; j < outputSize; ++j) {
                    file >> weightsHidden2Output[i][j];
                }
            }

            file.close();
        } else {
            cerr << "Не удалось открыть файл с весами." << endl;
        }
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
};


int main() {
    setlocale(LC_ALL,"RU"); //Для русского языка в консоли

    int inputSize = 4;
    int hiddenSize1 = 6;  // Новый размер первого скрытого слоя
    int hiddenSize2 = 2;  // Новый размер второго скрытого слоя
    int outputSize = 1;

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    neuralNet.loadWeightsFromFile("weights.txt");

    int N;
    cout << "Введите количество примеров: ";
    if (!(cin >> N) || N < 0) {
        cerr << "Некорректное количество примеров." << endl;
        return 1;
    }

    // Для каждого примера сеть принимает четыре входных значения.
    double (*newInputs)[4] = new double[N][4]();
    for (int i = 0; i < N; ++i) {
        cout << "Введите значения для примера " << i + 1 << ": ";
        for (int j = 0; j < inputSize; ++j) {
            if (!(cin >> newInputs[i][j])) {
                cerr << "Некорректное входное значение." << endl;
                delete[] newInputs;
                return 1;
            }
        }
    }
    double* prediction = new double[outputSize];
    for (int i = 0; i < N; ++i) {
        const double* newInput = newInputs[i];
        neuralNet.forward(newInput, prediction);
        cout << "Вход: " << newInput[0] << ',' << newInput[1] << ',' << newInput[2] << ',' << newInput[3]
             << " Предсказание: " << prediction[0] << endl;
    }
    delete[] prediction;
    delete[] newInputs;
    return 0;
}
