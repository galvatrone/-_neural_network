#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <cstdio>
#include <string>
#include <iomanip>

using namespace std;

// Моё любимое число. Один и тот же seed даёт одинаковые начальные веса
// и одинаковое разбиение на train/test при каждом запуске.
const unsigned int SEED = 31415962;

const string DATA_FILE = "ocrData.txt";               // буква + 16 признаков в строке
const string WEIGHTS_FILE = "letters_weights.txt";    // веса этой задачи 
const string CONFUSION_FILE = "confusion_matrix.csv";
const string SCALING_FILE = "letters_scaling.txt";   // min и max признаков из train, нужны nn_main

const int FEATURE_COUNT = 16;     // признаков у каждой буквы
const int CLASS_COUNT = 26;       // буквы A..Z
const double TEST_SHARE = 0.2;    // 80% train, 20% test

const double LEARNING_RATE = 0.1;
const int EPOCHS = 500;

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
            initializeRandomWeights();
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

    void initializeRandomWeights() {
        srand(SEED);

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

        // Смещения (bias) — тем же способом
        for (int i = 0; i < hiddenSize1; ++i) biasHidden1[i] = (rand() % 200) / 100.0 - 1.0;
        for (int i = 0; i < hiddenSize2; ++i) biasHidden2[i] = (rand() % 200) / 100.0 - 1.0;
        for (int i = 0; i < outputSize; ++i) biasOutput[i] = (rand() % 200) / 100.0 - 1.0;
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

            // Каждая матрица и каждый массив смещений — на своей строке,
            // порядок как в loadWeightsFromFile.
            writeMatrix(file, weightsInputHidden1, inputSize, hiddenSize1);
            writeMatrix(file, weightsHidden1Hidden2, hiddenSize1, hiddenSize2);
            writeMatrix(file, weightsHidden2Output, hiddenSize2, outputSize);
            writeMatrix(file, &biasHidden1, 1, hiddenSize1);
            writeMatrix(file, &biasHidden2, 1, hiddenSize2);
            writeMatrix(file, &biasOutput, 1, outputSize);

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
                hiddenInputs1[i] = biasHidden1[i];
                for (int j = 0; j < inputSize; ++j) {
                    hiddenInputs1[i] += input[j] * weightsInputHidden1[j][i];
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

            // Обновление смещений: у bias вход всегда равен 1
            for (int i = 0; i < hiddenSize1; ++i) biasHidden1[i] += learningRate * hiddenDeltas1[i];
            for (int i = 0; i < hiddenSize2; ++i) biasHidden2[i] += learningRate * hiddenDeltas2[i];
            for (int i = 0; i < outputSize; ++i) biasOutput[i] += learningRate * outputDeltas[i];

            // Суммарная ошибка
            for (int i = 0; i < outputSize; ++i) {
                totalError += abs(outputErrors[i]);
            }
        }

        // Вывод ошибки каждые 10 эпох
        if (epoch % 10 == 0) {
            cout << "Эпоха: " << epoch << ", Ошибка: " << totalError << endl;
            saveWeightsToFile(WEIGHTS_FILE);
        }

        // Если ошибка достаточно мала, завершаем обучение
        if (totalError < 0.01) {
            cout << "Обучение завершено. Эпоха: " << epoch << ", Ошибка: " << totalError << endl;
            break;
        }
        // Сохранение весов после каждой эпохи
        //saveWeightsToFile(WEIGHTS_FILE);
    }
    delete[] buffer;
}


private:

    static void writeMatrix(ofstream& file, double* const* matrix, int rows, int columns) {
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

    double* biasHidden1;
    double* biasHidden2;
    double* biasOutput;
};


// Читает файл с буквами: в каждой строке буква и FEATURE_COUNT целых признаков
// через запятую, например "T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8".
// Возвращает количество примеров или -1 при ошибке.
int loadDataset(const string& filename, double (*&features)[FEATURE_COUNT], int*& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл с данными: " << filename << endl;
        return -1;
    }

    // Первый проход: считаем строки, чтобы выделить массивы нужного размера.
    int count = 0;
    string line;
    while (getline(file, line)) {
        if (!line.empty() && line != "\r") ++count;
    }
    file.clear();
    file.seekg(0);

    features = new double[count][FEATURE_COUNT];
    labels = new int[count];

    // Второй проход: разбираем строки.
    for (int n = 0; n < count; ++n) {
        char letter, comma;
        if (!(file >> letter) || letter < 'A' || letter > 'Z') {
            cerr << "Строка " << n + 1 << ": ожидалась буква A-Z." << endl;
            delete[] features;
            delete[] labels;
            return -1;
        }
        labels[n] = letter - 'A';   // кодирование метки: A=0, B=1, ..., Z=25
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            if (!(file >> comma >> features[n][j]) || comma != ',') {
                cerr << "Строка " << n + 1 << ": ожидалось " << FEATURE_COUNT << " чисел через запятую." << endl;
                delete[] features;
                delete[] labels;
                return -1;
            }
        }
    }
    return count;
}

// Стратифицированное разбиение: для каждой буквы отдельно перемешиваем её примеры
// и отправляем долю testShare в test, остальное в train. Так доли букв в train
// и test такие же, как во всём наборе. Перемешивание использует SEED.
void stratifiedSplit(const int* labels, int count, double testShare,
                     int* trainIndex, int& trainCount, int* testIndex, int& testCount) {
    srand(SEED);
    trainCount = 0;
    testCount = 0;
    int* classIndex = new int[count];

    for (int c = 0; c < CLASS_COUNT; ++c) {
        // Номера всех примеров буквы c
        int classSize = 0;
        for (int n = 0; n < count; ++n) {
            if (labels[n] == c) classIndex[classSize++] = n;
        }
        // Перемешивание Фишера–Йетса
        for (int i = classSize - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            int temp = classIndex[i]; classIndex[i] = classIndex[j]; classIndex[j] = temp;
        }
        int classTest = static_cast<int>(classSize * testShare + 0.5);
        for (int i = 0; i < classSize; ++i) {
            if (i < classTest) testIndex[testCount++] = classIndex[i];
            else trainIndex[trainCount++] = classIndex[i];
        }
    }

    // После разбиения train идёт по буквам подряд (все A, потом все B...).
    // Перемешиваем, иначе сеть будет учить буквы по очереди и забывать прошлые.
    for (int i = trainCount - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int temp = trainIndex[i]; trainIndex[i] = trainIndex[j]; trainIndex[j] = temp;
    }
    delete[] classIndex;
}

// Масштабирование в [0, 1]: min и max каждого признака ищем ТОЛЬКО по train,
// а применяем ко всем примерам (и train, и test).
// Найденные min и max возвращаются через minValues и maxValues, чтобы их можно было
// сохранить и так же масштабировать новые буквы в nn_main.
void scaleFeatures(double (*features)[FEATURE_COUNT], int count, const int* trainIndex, int trainCount,
                   double* minValues, double* maxValues) {
    for (int j = 0; j < FEATURE_COUNT; ++j) {
        minValues[j] = features[trainIndex[0]][j];
        maxValues[j] = features[trainIndex[0]][j];
    }
    for (int i = 0; i < trainCount; ++i) {
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            double value = features[trainIndex[i]][j];
            if (value < minValues[j]) minValues[j] = value;
            if (value > maxValues[j]) maxValues[j] = value;
        }
    }
    for (int n = 0; n < count; ++n) {
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            double range = maxValues[j] - minValues[j];
            features[n][j] = range > 0 ? (features[n][j] - minValues[j]) / range : 0.0;
        }
    }
}



// Сохраняет min (первая строка) и max (вторая строка) каждого признака.
bool saveScaling(const double* minValues, const double* maxValues, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Не удалось создать файл: " << filename << endl;
        return false;
    }
    for (int j = 0; j < FEATURE_COUNT; ++j) file << minValues[j] << ' ';
    file << '\n';
    for (int j = 0; j < FEATURE_COUNT; ++j) file << maxValues[j] << ' ';
    file << '\n';
    return !file.fail();
}


// Номер выхода с наибольшим значением — это и есть предсказанная буква.
int predictClass(NeuralNetwork& neuralNet, const double* input, double* outputs) {
    neuralNet.forward(input, outputs);
    int best = 0;
    for (int i = 1; i < CLASS_COUNT; ++i) {
        if (outputs[i] > outputs[best]) best = i;
    }
    return best;
}

// Точность на выбранных примерах. Если confusion не nullptr, заполняет матрицу
// ошибок: строка — настоящая буква, столбец — предсказанная.
double accuracy(NeuralNetwork& neuralNet, double (*features)[FEATURE_COUNT], const int* labels,
                const int* index, int count, int (*confusion)[CLASS_COUNT]) {
    double outputs[CLASS_COUNT];
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        int n = index[i];
        int predicted = predictClass(neuralNet, features[n], outputs);
        if (predicted == labels[n]) ++correct;
        if (confusion != nullptr) ++confusion[labels[n]][predicted];
    }
    return count > 0 ? static_cast<double>(correct) / count : 0.0;
}

void printConfusionMatrix(int (*confusion)[CLASS_COUNT]) {
    cout << "Матрица ошибок на test (строка — настоящая буква, столбец — предсказанная):" << endl;
    cout << "   ";
    for (int j = 0; j < CLASS_COUNT; ++j) cout << setw(4) << static_cast<char>('A' + j);
    cout << endl;
    for (int i = 0; i < CLASS_COUNT; ++i) {
        cout << setw(3) << static_cast<char>('A' + i);
        for (int j = 0; j < CLASS_COUNT; ++j) cout << setw(4) << confusion[i][j];
        cout << endl;
    }
}

bool saveConfusionMatrix(int (*confusion)[CLASS_COUNT], const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Не удалось создать файл: " << filename << endl;
        return false;
    }
    file << "true\\predicted";
    for (int j = 0; j < CLASS_COUNT; ++j) file << ',' << static_cast<char>('A' + j);
    file << '\n';
    for (int i = 0; i < CLASS_COUNT; ++i) {
        file << static_cast<char>('A' + i);
        for (int j = 0; j < CLASS_COUNT; ++j) file << ',' << confusion[i][j];
        file << '\n';
    }
    return !file.fail();
}

// Самые частые путаницы: наибольшие числа вне диагонали матрицы ошибок.
void printTopConfusions(int (*confusion)[CLASS_COUNT], int top) {
    cout << "Самые частые ошибки (настоящая -> предсказанная: сколько раз):" << endl;
    bool used[CLASS_COUNT][CLASS_COUNT] = {};
    for (int k = 0; k < top; ++k) {
        int bestI = -1, bestJ = -1;
        for (int i = 0; i < CLASS_COUNT; ++i) {
            for (int j = 0; j < CLASS_COUNT; ++j) {
                if (i == j || used[i][j] || confusion[i][j] == 0) continue;
                if (bestI < 0 || confusion[i][j] > confusion[bestI][bestJ]) { bestI = i; bestJ = j; }
            }
        }
        if (bestI < 0) break;
        used[bestI][bestJ] = true;
        cout << "  " << static_cast<char>('A' + bestI) << " -> " << static_cast<char>('A' + bestJ)
             << ": " << confusion[bestI][bestJ] << endl;
    }
}


int main() {
    int inputSize = FEATURE_COUNT;   // 16 признаков
    int hiddenSize1 = 32;            // по 2 нейрона на каждый входной признак
    int hiddenSize2 = 52;            // по 2 нейрона на каждую букву
    int outputSize = CLASS_COUNT;    // 26 букв

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);

    // Если файла весов нет — начинаем со случайных весов из конструктора.
    // Если файл есть, но испорчен — останавливаемся, чтобы не затереть его.
    ifstream existingWeights(WEIGHTS_FILE);
    if (existingWeights.is_open()) {
        existingWeights.close();
        if (!neuralNet.loadWeightsFromFile(WEIGHTS_FILE)) {
            return 1;
        }
        cout << "Продолжаем обучение с весов из " << WEIGHTS_FILE << endl;
    } else {
        cout << "Файл " << WEIGHTS_FILE << " не найден, начинаем со случайных весов (seed " << SEED << ")" << endl;
    }

    // Загрузка примеров из файла
    double (*features)[FEATURE_COUNT] = nullptr;
    int* labels = nullptr;
    int count = loadDataset(DATA_FILE, features, labels);
    if (count <= 0) {
        return 1;
    }

    // Разбиение 80/20 со стратификацией и масштабирование по train
    int* trainIndex = new int[count];
    int* testIndex = new int[count];
    int trainCount = 0, testCount = 0;
    stratifiedSplit(labels, count, TEST_SHARE, trainIndex, trainCount, testIndex, testCount);
    double minValues[FEATURE_COUNT];
    double maxValues[FEATURE_COUNT];
    scaleFeatures(features, count, trainIndex, trainCount, minValues, maxValues);
    if (!saveScaling(minValues, maxValues, SCALING_FILE)) {
        return 1;
    }
    cout << "Примеров: " << count << ", train: " << trainCount << ", test: " << testCount << endl;

    // Целевые значения one-hot: у правильной буквы 1, у остальных 0
    double (*targets)[CLASS_COUNT] = new double[count][CLASS_COUNT]();
    for (int n = 0; n < count; ++n) {
        targets[n][labels[n]] = 1.0;
    }

    // Обучение сети только на train
    const double** inputRows = new const double*[trainCount];
    const double** targetRows = new const double*[trainCount];
    for (int i = 0; i < trainCount; ++i) {
        inputRows[i] = features[trainIndex[i]];
        targetRows[i] = targets[trainIndex[i]];
    }
    neuralNet.train(inputRows, targetRows, trainCount, LEARNING_RATE, EPOCHS);

    // Оценка сети
    int (*confusion)[CLASS_COUNT] = new int[CLASS_COUNT][CLASS_COUNT]();
    double trainAccuracy = accuracy(neuralNet, features, labels, trainIndex, trainCount, nullptr);
    double testAccuracy = accuracy(neuralNet, features, labels, testIndex, testCount, confusion);

    cout << fixed << setprecision(2);
    cout << "Точность на train: " << trainAccuracy * 100 << "%" << endl;
    cout << "Точность на test:  " << testAccuracy * 100 << "%" << endl;
    printConfusionMatrix(confusion);
    printTopConfusions(confusion, 10);

    bool saved = neuralNet.saveWeightsToFile(WEIGHTS_FILE);
    saved = saveConfusionMatrix(confusion, CONFUSION_FILE) && saved;

    delete[] confusion;
    delete[] inputRows;
    delete[] targetRows;
    delete[] targets;
    delete[] trainIndex;
    delete[] testIndex;
    delete[] features;
    delete[] labels;
    return saved ? 0 : 1;
}
