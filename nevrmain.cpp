#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

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

    vector<double> forward(const vector<double>& inputs) {
        vector<double> hiddenInputs1(hiddenSize1);
        vector<double> hiddenOutputs1(hiddenSize1);
        vector<double> hiddenInputs2(hiddenSize2);
        vector<double> hiddenOutputs2(hiddenSize2);
        vector<double> finalInputs(outputSize);
        vector<double> finalOutputs(outputSize);

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

        return finalOutputs;
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


    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets,
           double learningRate = 0.9, int epochs = 1000000000) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (size_t example = 0; example < inputs.size(); ++example) {
            const vector<double>& input = inputs[example];
            const vector<double>& target = targets[example];

            // Прямой проход
            vector<double> hiddenInputs1(hiddenSize1);
            vector<double> hiddenOutputs1(hiddenSize1);
            vector<double> hiddenInputs2(hiddenSize2);
            vector<double> hiddenOutputs2(hiddenSize2);
            vector<double> finalInputs(outputSize);
            vector<double> finalOutputs(outputSize);

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
        if (totalError < 0.01) {
            cout << "Обучение завершено. Эпоха: " << epoch << ", Ошибка: " << totalError << endl;
            break;
        }
        // Сохранение весов после каждой эпохи
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
    int hiddenSize1 = 6;  // Новый размер первого скрытого слоя
    int hiddenSize2 = 2;  // Новый размер второго скрытого слоя
    int outputSize = 1;

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    neuralNet.loadWeightsFromFile("weights.txt");
    // Пример входных данных и целевых значений
    vector<vector<double>> inputs = {{0, 0,1,1}, {1, 1,0,0}, {1, 0,0,1}, {1,0, 1,0},{0,1,0,1},{0,1,1,0},{0,1,1,1},{0,1,0,0},{0,1,2,3},{0,2,2,1}};
    vector<vector<double>> targets = {{1}, {1}, {1}, {0},{0},{1},{2},{1},{0},{1}};

    // Обучение сети
    neuralNet.train(inputs, targets);

    // Оценка сети
    cout << "Результаты после обучения:" << endl;
    for (const auto& input : inputs) {
        vector<double> predictions = neuralNet.forward(input);
        cout << "Вход: " << input[0] << ", " << input[1] << ','<<input[2] << ", " << input[3] <<  " Выход: " << predictions[0] << endl;
    }
    neuralNet.saveWeightsToFile("weights.txt");
    return 0;
}
