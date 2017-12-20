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
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//b36cdd66cc930db82e0861639504c0eb.svg?invert_in_darkmode" align=middle width=413.50155pt height=50.226165pt/></p>

Функция активации на скрытом слое тангес:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//ed95e6fc19764479cbcac06b6293beab.svg?invert_in_darkmode" align=middle width=117.602925pt height=37.147275pt/></p>

Функция активации на втором слое softmax:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//f8ce48e4708e476899413ca68cba9971.svg?invert_in_darkmode" align=middle width=126.424485pt height=40.62036pt/></p>

### Алгоритм
1. Инициализируем веса значениями из диапазона [0, 0.5]
2. Пока количество проходов < max_epoch делаем:

	Для всех картинок от 1 до number_train_images
	+ Подаем на вход x, суммируем cигналы на скрытом слое <img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//4103d3e79fea5164061ddeaa7684d372.svg?invert_in_darkmode" align=middle width=8.1885045pt height=14.10255pt/> применяем функцию активации <img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//fa735bd3d5226cf9d1545f8e82e7c15b.svg?invert_in_darkmode" align=middle width=8.1885045pt height=14.10255pt/>

	+ Для каждого выходного нейрона суммируем взвешенные входящие сигналы <img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//2bdeadcb4b376398c4ec45103ab9af7f.svg?invert_in_darkmode" align=middle width=8.1885045pt height=14.10255pt/> применяем функцию1 активации <img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//83781cdfb4d517363b69c2c87a038940.svg?invert_in_darkmode" align=middle width=8.1885045pt height=14.10255pt/>

	+ Считаем градиенты функции ошибки:

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//33a5033e2a26ab9a0fdd1e5448f360f8.svg?invert_in_darkmode" align=middle width=125.15745pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//f31d0c7bd21cc5253aa7b195bbef0523.svg?invert_in_darkmode" align=middle width=96.984855pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//73d12a65bbf307b427bcf40f4e138867.svg?invert_in_darkmode" align=middle width=73.424175pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//902c32840fc99fdf2a8963a1f8f77d64.svg?invert_in_darkmode" align=middle width=186.6447pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//c8a6c938ccd9641d27cb2404ac4cf16c.svg?invert_in_darkmode" align=middle width=121.93533pt height=36.27789pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//e5fe7703c9325ead4c97db212f417354.svg?invert_in_darkmode" align=middle width=427.5579pt height=50.226165pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//cb94e20f4bd01e87c2e1c95ef063383e.svg?invert_in_darkmode" align=middle width=180.84165pt height=50.226165pt/></p>

 	 В случае гиперболического тангенса: 

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//a2d1f5c4c1cfd4eb7b679350d8aeb85d.svg?invert_in_darkmode" align=middle width=160.38198pt height=36.27789pt/></p>

 	 Тогда 

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//e047b5019a674502886d7230d6dfb0ad.svg?invert_in_darkmode" align=middle width=87.65658pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//abe9d2643906994afdf0355a0f9f34d3.svg?invert_in_darkmode" align=middle width=86.176365pt height=36.27789pt/></p>
	+ Корректируем веса в соответствии с градиентами функции ошибки:

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//e3b96ae350a7aaac265e6ce1ad594b5a.svg?invert_in_darkmode" align=middle width=155.663805pt height=36.27789pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/dev_vroy/tex//b1af6355d19d273f5321c89b4e081f8d.svg?invert_in_darkmode" align=middle width=160.22424pt height=38.51529pt/></p>

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