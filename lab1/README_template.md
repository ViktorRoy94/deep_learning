# Лабораторная работа №1

Выполнил:
 - Рой Виктор
 - ННГУ, ф-т ИТММ, каф. МО ЭВМ, группа 381603м4

## Задание
 - Изучить метод обратного распространения ошибки;
 - Вывести математические формулы для вычисления градиентов функции ошибки по параметрам
нейронной сети и формул коррекции весов;
 - Спроектировать и разработать программную реализацию;
 - Подготовить отчет по проделанной работе. 
 
## Запуск решения
 - Установить python3 c библиотекой Numpy;
 - Запустить файл main.py с аргументами 
 	- '-h' помощь 
 	- '-n' количество тренировочных изображений
 	- '-t' количество тестовых изображений
 	- '-s' количество скрытых слоев
 	- '-l' скорость обучения

Пример запуска: python main.py -n 10000 -t 1000 -s 200 -l 0.005

## Метод обратного распространения ошибки
Функция ошибки кросс-энтропия:
$$E=-\sum_{j=1}^{N_o}t_jlog(y_j) = -\sum_{j=1}^{N_o}t_jlog(f(\sum_{s=1}^{N_s}w_s_jf(\sum_{i=1}^{N_i}w_i_sx_i)))$$

Функция активации на скрытом слое тангес:
$$\phi(y_j)={{e^{2y_j}-1}\over{e^{2y_j}+1}}$$

Функция активации на втором слое softmax:
$$f(y_j)={{e^{y_j}}\over\sum_{j=1}^{n}e^{y_j}}$$

### Алгоритм
1. Инициализируем веса значениями из диапазона [0, 0.5]
2. Пока количество проходов < max_epoch делаем:

	Для всех картинок от 1 до number_train_images
	+ Подаем на вход x, суммируем cигналы на скрытом слое $\text{z_s}=w_0_s+\sum_{i}^{N_i}{w_i_s}x_i,$ применяем функцию активации $\text{v_s}=\phi(z_s)$

	+ Для каждого выходного нейрона суммируем взвешенные входящие сигналы $\text{y_j}=w_0_j+\sum_{s}^{N_s}{w_s_j}{f(z_s)},$ применяем функцию активации $\text{u_j}=f(y_j)$

	+ Считаем градиенты функции ошибки:

	$${{\partial{E}}\over{\partial{w_s_j}}}={{\partial{E}}\over{\partial{y_j}}}{{\partial{y_j}}\over{\partial{w_s_j}}}$$

	$${{\partial{E}}\over{\partial{y_j}}}=u_j-t_j$$

	$${{\partial{y_j}}\over{\partial{w_s_j}}}=v_s$$

	$${{\partial{E}}\over{\partial{w_s_j}}}=(u_j-t_j)v_s={\delta_j{v_s}}$$

	$${{\partial{E}}\over{\partial{w_i_s}}}={{\partial{E}}\over{\partial{z_s}}}{{\partial{z_s}}\over{\partial{w_i_s}}}$$

	$${{\partial{E}}\over{\partial{z_s}}}=\sum_{j=1}^{N_o}{{\partial{E}}\over{\partial{y_j}}}{{\partial{y_j}}\over{\partial{v_s}}}{{\partial{f}}\over{\partial{z_s}}}={{\partial{f}}\over{\partial{z_s}}}\sum_{j=1}^{N_o}{{\partial{E}}\over{\partial{y_j}}}{{\partial{y_j}}\over{\partial{v_s}}}={{\partial{f}}\over{\partial{z_s}}}(\sum_{j=1}^{N_o}{\delta_j^2w_s_j^2})$$

	$${{\partial{E}}\over{\partial{w_i_s}}}={{\partial{f}}\over{\partial{z_s}}}(\sum_{j=1}^{N_o}{\delta_j^2w_s_j^2}){x_i}$$

 	 В случае гиперболического тангенса: 

	$${{\partial{f}}\over{\partial{z_s}}}=(1-v_s)(1+v_s)$$

 	 Тогда 

	$${{\partial{E}}\over{\partial{w_s_j}}}={\delta_j{v_s}}$$

	$${{\partial{E}}\over{\partial{w_i_s}}}={{\delta_s}{x_i}}$$
	+ Корректируем веса в соответствии с градиентами функции ошибки:

	$${w_i_s^{n+1}=w_i_s^{n}-\eta{{\partial{E}}\over{\partial{w_i_s}}}}$$

	$${w_s_j^{n+1}=w_s_j^{n}-\eta{{\partial{E}}\over{\partial{w_s_j}}}}$$

## Реализация 
Программа написана на Python 3.6. В рамках лабораторной работы был создан класс NeuraNetwork.py, в котором реализованы следущие методы:

 - initializeWeights() - инициализация весов случайными значениями из диапазона [0, 0.5];
 - train(self, x_values, t_values, maxEpochs, learnRate, crossError) - обучение сети с помощью метода обратного распространения ошибки;
 - computeOutputs(self, xValues) - расчет значений на выходе сети;
 - computeGradient(self, t_values, oGrads, hGrads) - расчет градиентов для обновления весов перед следующим шагом алгоритма;
 - updateWeightsAndBiases(self, learnRate, hGrads, oGrads) - обновление весов сетки;
 - crossEntropyError(self, x_values, t_values) - расчет величины кросс-энтропии;
 - accuracy(self, x_values, t_values) - расчет ошибки в натренированной нейронной сети.
## Результаты экспериментов

| Число нейронов скрытого слоя | К-во эпох | Точность тренировочная | Точность тестовая |
| :---: | :---:  |   :---:     |   :---:   |
|  100  | 14     |    0.9974   |  0.9725   |
|  200  | 17     |    0.9982   |  0.9837   |
|  300  | 22     |    0.9994   |  0.9821   |