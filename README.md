# -_neural_network
C++ neural network implementation with customizable architecture. Train and evaluate on diverse datasets. Save/load weights, ensuring flexibility. Ideal for machine learning experiments. Contributions welcome. MIT License.

Welcome to Galvatrone, a C++ neural network implementation with customizable architecture.

## Description

This project provides a flexible neural network framework designed for machine learning experiments and practical applications. The architecture allows customization of input sizes, hidden layer configurations, and output sizes. Train the network using backpropagation, save/load weights, and adapt it to various problem domains.

## Сборка и запуск

Нейросеть использует динамические массивы `new[]` / `delete[]`, без `std::vector`.
Структура слоёв, сигмоида, алгоритм обучения и текстовый формат `weights.txt` сохранены.

Имена файлов в текущем проекте:

- `NN_main.cpp` — обучение, сохранение весов и вывод результатов.
- `NN_training.cpp` — загрузка весов и предсказания для введённых примеров.

```bash
g++ -std=c++11 -Wall -Wextra -Wpedantic NN_main.cpp -o NN_main
g++ -std=c++11 -Wall -Wextra -Wpedantic NN_training.cpp -o NN_training
./NN_main
./NN_training
```

Обе программы используют `weights.txt` в текущем рабочем каталоге.
Программа предсказаний запрашивает количество примеров и четыре числа для каждого.
В демонстрации обучения сохранены исходные параметры: слои 4 → 6 → 2 → 1,
скорость обучения 0.9, максимум 1 000 000 000 эпох и остановка при ошибке меньше 0.01.

## Использование класса

`NeuralNetwork` принимает размеры слоёв через конструктор и освобождает свои массивы
в деструкторе. Копирование запрещено, чтобы два объекта не владели одной памятью.

```cpp
double input[] = {1, 0, 0, 1};
double result[1];
neuralNet.forward(input, result);

const double* inputs[] = {input};
double target[] = {1};
const double* targets[] = {target};
neuralNet.train(inputs, targets, 1, 0.9, 100);
```

В `forward` вызывающий код выделяет результат длиной `outputSize`.
В `train` передаются массивы указателей на примеры и цели, а затем число примеров;
длина каждой строки должна соответствовать `inputSize` или `outputSize`.
Метод `train` доступен в версии класса для обучения, в `NN_main.cpp`.

## Проверки

```bash
python tests/run_tests.py
```

Проверки компилируют обе программы с AddressSanitizer и UndefinedBehaviorSanitizer,
проверяют вычисления, короткое обучение, формат весов, освобождение памяти и консольный ввод.
Нужны Python 3 и g++ с поддержкой санитайзеров. Полный демонстрационный цикл до миллиарда
эпох при проверке не запускается.
