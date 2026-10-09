#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>

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
            initializeRandomWeights();
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

    void initializeRandomWeights() {
        srand(static_cast<unsigned int>(time(nullptr)));

        // Инициализация весов в пределах [-1, 1] для первого слоя
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize1; ++j) {
                weightsInputHidden1[i][j] = (rand() % 200) / 100.0 - 1.0;
            }
        }

        // Инициализация весов в пределах [-1, 1] для второго слоя
        for (int i = 0; i < hiddenSize1; ++i) {
            for (int j = 0; j < hiddenSize2; ++j) {
                weightsHidden1Hidden2[i][j] = (rand() % 200) / 100.0 - 1.0;
            }
        }

        // Инициализация весов в пределах [-1, 1] для выходного слоя
        for (int i = 0; i < hiddenSize2; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weightsHidden2Output[i][j] = (rand() % 200) / 100.0 - 1.0;
            }
        }
    }


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

    void saveWeightsToFile(const string& filename) {
        ofstream file(filename);
        if (file.is_open()) {
            // Сохранение весов для первого скрытого слоя
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < hiddenSize1; ++j) {
                    file << weightsInputHidden1[i][j] << " ";
                }
            }

            // Сохранение весов для второго скрытого слоя
            for (int i = 0; i < hiddenSize1; ++i) {
                for (int j = 0; j < hiddenSize2; ++j) {
                    file << weightsHidden1Hidden2[i][j] << " ";
                }
            }

            // Сохранение весов для выходного слоя
            for (int i = 0; i < hiddenSize2; ++i) {
                for (int j = 0; j < outputSize; ++j) {
                    file << weightsHidden2Output[i][j] << " ";
                }
            }

            file.close();
        }
    }


    void train(const double* const* inputs, const double* const* targets, int exampleCount,
           double learningRate = 0.9, int epochs = 1000000000) {
    // Рабочие массивы выделяются один раз на всё обучение.
    double* buffer = new double[4 * static_cast<size_t>(hiddenSize1) + 4 * static_cast<size_t>(hiddenSize2) + 4 * static_cast<size_t>(outputSize)];
    double* hiddenInputs1 = buffer;
    double* hiddenOutputs1 = hiddenInputs1 + hiddenSize1;
    double* hiddenInputs2 = hiddenOutputs1 + hiddenSize1;
    double* hiddenOutputs2 = hiddenInputs2 + hiddenSize2;
    double* finalInputs = hiddenOutputs2 + hiddenSize2;
    double* finalOutputs = finalInputs + outputSize;
    double* outputErrors = finalOutputs + outputSize;
    double* outputDeltas = outputErrors + outputSize;
    double* hiddenErrors2 = outputDeltas + outputSize;
    double* hiddenDeltas2 = hiddenErrors2 + hiddenSize2;
    double* hiddenErrors1 = hiddenDeltas2 + hiddenSize2;
    double* hiddenDeltas1 = hiddenErrors1 + hiddenSize1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (int example = 0; example < exampleCount; ++example) {
            const double* input = inputs[example];
            const double* target = targets[example];

            // Прямой проход

            // Прямой проход через первый скрытый слой
            for (int i = 0; i < hiddenSize1; ++i) {
                hiddenInputs1[i] = 0;
                for (int j = 0; j < inputSize; ++j) {
                    hiddenInputs1[i] += input[j] * weightsInputHidden1[j][i];
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

            // Обратное распространение ошибки

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

            // Обновление весов
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

            // Суммарная ошибка
            for (int i = 0; i < outputSize; ++i) {
                totalError += abs(outputErrors[i]);
            }
        }

        // Вывод ошибки каждые 1000 эпох
        if (epoch % 10000 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << totalError << endl;
            saveWeightsToFile("weights.txt");
        }

        // Если ошибка достаточно мала, завершаем обучение
        if (totalError < 0.11) {
            cout << "Обучение завершено. Эпоха: " << epoch << ", Ошибка: " << totalError << endl;
            break;
        }
        // Сохранение весов после каждой эпохи
        //saveWeightsToFile("weights.txt");
    }
    delete[] buffer;
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
    int inputSize = 4;
    int hiddenSize1 = 6;  // Новый размер первого скрытого слоя
    int hiddenSize2 = 2;  // Новый размер второго скрытого слоя
    int outputSize = 1;

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    neuralNet.loadWeightsFromFile("weights.txt");
    // Пример входных данных и целевых значений
    double (*inputs)[4] = new double[10][4] {
    {0,0,1,1}, {1,1,0,0}, {1,0,0,1}, {1,0,1,0}, {0,1,0,1},
    {0,1,1,0}, {0,1,1,1}, {0,1,0,0}, {0,0,0,1}, {0,1,0,1}};
    double (*targets)[1] = new double[10][1] {{1}, {1}, {1}, {0},{0},{1},{1},{1},{1},{0}};

    // Обучение сети
    const double* inputRows[10];
    const double* targetRows[10];
    for (int i = 0; i < 10; ++i) {
        inputRows[i] = inputs[i];
        targetRows[i] = targets[i];
    }
    neuralNet.train(inputRows, targetRows, 10);

    // Оценка сети
    cout << "Результаты после обучения:" << endl;
    double* predictions = new double[outputSize];
    for (int i = 0; i < 10; ++i) {
        const double* input = inputs[i];
        neuralNet.forward(input, predictions);
        cout << "Вход: " << input[0] << ", " << input[1] << ','<<input[2] << ", " << input[3] <<  " Выход: " << predictions[0] << endl;
    }
    neuralNet.saveWeightsToFile("weights.txt");
    delete[] predictions;
    delete[] inputs;
    delete[] targets;
    return 0;
}
