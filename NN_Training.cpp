#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <random>
#include <limits>
#include <cstdio>
#include <string>

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

    // Заполняет все веса случайными числами из [-1, 1).
    // Генератор создаётся один раз на программу, поэтому сети, созданные подряд,
    // получают разные веса (а не одинаковые, как было с srand(time) в одну секунду).
    void initializeRandomWeights() {
        static mt19937 generator = makeGenerator();
        uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize1; ++j)
                weightsInputHidden1[i][j] = distribution(generator);
        for (int i = 0; i < hiddenSize1; ++i)
            for (int j = 0; j < hiddenSize2; ++j)
                weightsHidden1Hidden2[i][j] = distribution(generator);
        for (int i = 0; i < hiddenSize2; ++i)
            for (int j = 0; j < outputSize; ++j)
                weightsHidden2Output[i][j] = distribution(generator);
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

    // Читает веса во временный массив и копирует их в сеть только после полной
    // проверки: при любой ошибке текущие веса остаются без изменений.
    bool loadWeightsFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Не удалось открыть файл с весами: " << filename << endl;
            return false;
        }

        const size_t count = weightCount();
        double* values = new double[count];
        for (size_t k = 0; k < count; ++k) {
            if (!(file >> values[k]) || !isfinite(values[k])) {
                cerr << "Файл весов повреждён или содержит меньше " << count << " чисел: " << filename << endl;
                delete[] values;
                return false;
            }
        }
        double extra;
        if (file >> extra || !file.eof()) {
            cerr << "В файле весов больше " << count << " чисел или лишние данные"
                 << " (другая архитектура сети?): " << filename << endl;
            delete[] values;
            return false;
        }

        size_t k = 0;
        // Порядок: вход → скрытый слой 1, скрытый 1 → скрытый 2, скрытый 2 → выход.
        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < hiddenSize1; ++j)
                weightsInputHidden1[i][j] = values[k++];
        for (int i = 0; i < hiddenSize1; ++i)
            for (int j = 0; j < hiddenSize2; ++j)
                weightsHidden1Hidden2[i][j] = values[k++];
        for (int i = 0; i < hiddenSize2; ++i)
            for (int j = 0; j < outputSize; ++j)
                weightsHidden2Output[i][j] = values[k++];

        delete[] values;
        return true;
    }

    // Количество весов в файле для текущей архитектуры.
    size_t weightCount() const {
        return static_cast<size_t>(inputSize) * hiddenSize1
             + static_cast<size_t>(hiddenSize1) * hiddenSize2
             + static_cast<size_t>(hiddenSize2) * outputSize;
    }

    // Пишет веса во временный файл и заменяет им старый только после успешной
    // записи: если программу прервут на середине, прежний файл не испортится.
    bool saveWeightsToFile(const string& filename) const {
        const string tempName = filename + ".tmp";
        {
            ofstream file(tempName);
            if (!file.is_open()) {
                cerr << "Не удалось создать файл весов: " << tempName << endl;
                return false;
            }
            // Достаточно цифр, чтобы при загрузке получить точно то же число double.
            file.precision(numeric_limits<double>::max_digits10);

            // Каждая матрица — на своей строке, порядок как в loadWeightsFromFile.
            writeMatrix(file, weightsInputHidden1, inputSize, hiddenSize1);
            writeMatrix(file, weightsHidden1Hidden2, hiddenSize1, hiddenSize2);
            writeMatrix(file, weightsHidden2Output, hiddenSize2, outputSize);

            file.close();
            if (file.fail()) {
                cerr << "Ошибка записи файла весов: " << tempName << endl;
                remove(tempName.c_str());
                return false;
            }
        }
        if (rename(tempName.c_str(), filename.c_str()) != 0) {
            // В Windows rename не заменяет существующий файл: удаляем и пробуем снова.
            remove(filename.c_str());
            if (rename(tempName.c_str(), filename.c_str()) != 0) {
                cerr << "Не удалось заменить файл весов: " << filename << endl;
                remove(tempName.c_str());
                return false;
            }
        }
        return true;
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
        if (totalError < 0.01) {
            cout << "Обучение завершено. Эпоха: " << epoch << ", Ошибка: " << totalError << endl;
            break;
        }
        // Сохранение весов после каждой эпохи
        //saveWeightsToFile("weights.txt");
    }
    delete[] buffer;
}


private:

    static mt19937 makeGenerator() {
        // random_device на некоторых старых MinGW детерминирован, поэтому
        // дополнительно подмешиваем текущее время.
        random_device device;
        seed_seq seed{device(), device(), static_cast<unsigned int>(time(nullptr))};
        return mt19937(seed);
    }

    static void writeMatrix(ofstream& file, double** matrix, int rows, int columns) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                file << matrix[i][j] << ' ';
            }
        }
        file << '\n';
    }

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
    // Если файла весов нет — начинаем со случайных весов из конструктора.
    // Если файл есть, но испорчен — останавливаемся, чтобы не затереть его.
    ifstream existingWeights("weights.txt");
    if (existingWeights.is_open()) {
        existingWeights.close();
        if (!neuralNet.loadWeightsFromFile("weights.txt")) {
            return 1;
        }
        cout << "Продолжаем обучение с весов из weights.txt" << endl;
    } else {
        cout << "Файл weights.txt не найден, начинаем со случайных весов" << endl;
    }
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
    const bool saved = neuralNet.saveWeightsToFile("weights.txt");
    delete[] predictions;
    delete[] inputs;
    delete[] targets;
    return saved ? 0 : 1;
}
