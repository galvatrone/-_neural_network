#include <iostream>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <string>
#include <clocale>


using namespace std;

const string WEIGHTS_FILE = "letters_weights.txt";    // веса, обученные NN_Training
const string SCALING_FILE = "letters_scaling.txt";    // min и max признаков из train

const int FEATURE_COUNT = 16;     // признаков у каждой буквы
const int CLASS_COUNT = 26;       // буквы A..Z

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize)
        : inputSize(inputSize), hiddenSize1(hiddenSize1), hiddenSize2(hiddenSize2), outputSize(outputSize),
          weightsInputHidden1(nullptr), weightsHidden1Hidden2(nullptr), weightsHidden2Output(nullptr),
          biasHidden1(nullptr), biasHidden2(nullptr), biasOutput(nullptr) {
        if (inputSize <= 0 || hiddenSize1 <= 0 || hiddenSize2 <= 0 || outputSize <= 0) {
            throw invalid_argument("Размеры слоёв должны быть положительными.");
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
            hiddenInputs1[i] = biasHidden1[i];
            for (int j = 0; j < inputSize; ++j) {
                hiddenInputs1[i] += inputs[j] * weightsInputHidden1[j][i];
            }
            hiddenOutputs1[i] = sigmoid(hiddenInputs1[i]);
        }

        // Прямой проход через второй скрытый слой
        for (int i = 0; i < hiddenSize2; ++i) {
            hiddenInputs2[i] = biasHidden2[i];
            for (int j = 0; j < hiddenSize1; ++j) {
                hiddenInputs2[i] += hiddenOutputs1[j] * weightsHidden1Hidden2[j][i];
            }
            hiddenOutputs2[i] = sigmoid(hiddenInputs2[i]);
        }

        // Прямой проход через выходной слой
        for (int i = 0; i < outputSize; ++i) {
            finalInputs[i] = biasOutput[i];
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
        // Порядок: три матрицы весов, затем три массива смещений.
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

    // Количество чисел в файле весов для текущей архитектуры (веса + смещения).
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


// Масштабирование как при обучении: min (первая строка) и max (вторая строка)
// каждого признака, найденные NN_Training на train.
bool loadScaling(double* minValues, double* maxValues, const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл масштабирования: " << filename << endl;
        return false;
    }
    for (int j = 0; j < FEATURE_COUNT; ++j) {
        if (!(file >> minValues[j])) {
            cerr << "Файл масштабирования повреждён: " << filename << endl;
            return false;
        }
    }
    for (int j = 0; j < FEATURE_COUNT; ++j) {
        if (!(file >> maxValues[j])) {
            cerr << "Файл масштабирования повреждён: " << filename << endl;
            return false;
        }
    }
    return true;
}


int main() {
    setlocale(LC_ALL,"RU"); //Для русского языка в консоли

    int inputSize = FEATURE_COUNT;   // 16 признаков
    int hiddenSize1 = 32;            // по 2 нейрона на каждый входной признак
    int hiddenSize2 = 52;            // по 2 нейрона на каждую букву
    int outputSize = CLASS_COUNT;    // 26 букв

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    // Без корректных весов предсказания бессмысленны.
    if (!neuralNet.loadWeightsFromFile(WEIGHTS_FILE)) {
        return 1;
    }
    double minValues[FEATURE_COUNT];
    double maxValues[FEATURE_COUNT];
    if (!loadScaling(minValues, maxValues, SCALING_FILE)) {
        return 1;
    }

    int N;
    cout << "Введите количество примеров: ";
    if (!(cin >> N) || N < 0) {
        cerr << "Некорректное количество примеров." << endl;
        return 1;
    }

    // Для каждого примера сеть принимает 16 признаков буквы (числа 0..15),
    // как в строке ocrData.txt без буквы.
    double (*newInputs)[FEATURE_COUNT] = new double[N][FEATURE_COUNT]();
    for (int i = 0; i < N; ++i) {
        cout << "Введите " << FEATURE_COUNT << " признаков для примера " << i + 1 << ": ";
        for (int j = 0; j < inputSize; ++j) {
            if (!(cin >> newInputs[i][j])) {
                cerr << "Некорректное входное значение." << endl;
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

        // Буква — выход с наибольшим значением
        int best = 0;
        for (int k = 1; k < outputSize; ++k) {
            if (prediction[k] > prediction[best]) best = k;
        }
        cout << "Вход:";
        for (int j = 0; j < FEATURE_COUNT; ++j) cout << (j ? "," : " ") << newInput[j];
        cout << " Предсказание: " << static_cast<char>('A' + best)
             << " (выход " << prediction[best] << ")" << endl;
    }
    delete[] prediction;
    delete[] newInputs;
    return 0;
}
