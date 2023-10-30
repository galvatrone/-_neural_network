#include <iostream>
#include <cmath>
//#include <cstdlib>
//#include <ctime>
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

      //  initializeRandomWeights();
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
    setlocale(LC_ALL,"RU"); //Для русского языка в консоли

    int inputSize = 4;
    int hiddenSize1 = 6;  // Новый размер первого скрытого слоя
    int hiddenSize2 = 2;  // Новый размер второго скрытого слоя
    int outputSize = 1;

    NeuralNetwork neuralNet(inputSize, hiddenSize1, hiddenSize2, outputSize);
    neuralNet.loadWeightsFromFile("weights.txt");

    // Пример использования обученной сети
    //vector<vector<double>> newInputs = {{1,0,0, 1}, {0, 0,0,0}, {0,1,1, 0}, {1,0,1, 0},{0,1,1,1},{0,0,1,1},{1,1,1,1}};  // Ваши новые входные данные
     // Задаем размер вектора
    int N;
    cout << "Введите размер вектора: ";
    cin >> N;

    // Создаем вектор с указанным размером и заполняем его нулями
    vector<vector<double>> newInputs(N, vector<double>(4, 0.0));

    // Ввод значений для каждого вектора
    for (int i = 0; i < N; ++i) {
        cout << "Введите значения для вектора " << i + 1 << ": ";
        for (int j = 0; j < 4; ++j) {
            cin >> newInputs[i][j];
        }
    }
    for (const auto& newInput : newInputs) {
        vector<double> prediction = neuralNet.forward(newInput);
        cout << "Вход: " << newInput[0] << ',' << newInput[1] <<','<<newInput[2]<<','<<newInput[3]<<" Предсказание: " << prediction[0] << endl;
    }
   // system("pause");
    return 0;
}
