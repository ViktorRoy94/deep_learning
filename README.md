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

## Метод обратноого распространения ошибки
Функция ошибки кросс-энтропия:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/b36cdd66cc930db82e0861639504c0eb.svg?invert_in_darkmode" align=middle width=413.50154999999995pt height=50.226165pt/></p>

Функция активации на скрытом слое тангес:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/ed95e6fc19764479cbcac06b6293beab.svg?invert_in_darkmode" align=middle width=117.602925pt height=37.147275pt/></p>

Функция активации на втором слое softmax:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/f8ce48e4708e476899413ca68cba9971.svg?invert_in_darkmode" align=middle width=126.424485pt height=40.62036pt/></p>

### Алгоритм
1. Инициализируем веса <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/2cf68af2a7f22008cd34c07d0a454c09.svg?invert_in_darkmode" align=middle width=23.445674999999998pt height=9.5433525pt/></p> и <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/2de931a49231aca051b8756cffd491b7.svg?invert_in_darkmode" align=middle width=24.899325pt height=11.780802pt/></p> значениями из [0,0.5]
2. Пока количество проходов < max_epoch делаем:

	Для всех картинок от 1 до number_train_images
	1. Подаем на вход <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/6781c6832157749f4454f52347a5a9fb.svg?invert_in_darkmode" align=middle width=14.045888999999999pt height=9.5433525pt/></p>, суммируем cигналы на скрытом слое <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/24e5debb20cdd9d2d9a3364b5f0e09b4.svg?invert_in_darkmode" align=middle width=147.64645499999997pt height=47.988764999999994pt/></p>, применяем функцию активации <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/f51a384170c325d917092dd626941950.svg?invert_in_darkmode" align=middle width=74.185815pt height=16.438356pt/></p>
    2. Для каждого выходного нейрона суммируем взвешенные входящие сигналы  $$\inline{y_j}=w_0_j+\sum_{s}^{N_s}{w_s_j}f({z_s})$$ и применяем функцию активации $$\inline{u_j}=f(y_j)$$
	3. Считаем градиенты функции ошибки:

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/33a5033e2a26ab9a0fdd1e5448f360f8.svg?invert_in_darkmode" align=middle width=125.15744999999998pt height=38.51529pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/f31d0c7bd21cc5253aa7b195bbef0523.svg?invert_in_darkmode" align=middle width=96.984855pt height=38.51529pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/73d12a65bbf307b427bcf40f4e138867.svg?invert_in_darkmode" align=middle width=73.42417499999999pt height=38.51529pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/902c32840fc99fdf2a8963a1f8f77d64.svg?invert_in_darkmode" align=middle width=186.64469999999997pt height=38.51529pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/c8a6c938ccd9641d27cb2404ac4cf16c.svg?invert_in_darkmode" align=middle width=121.93533pt height=36.27789pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/e5fe7703c9325ead4c97db212f417354.svg?invert_in_darkmode" align=middle width=427.55789999999996pt height=50.226165pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/cb94e20f4bd01e87c2e1c95ef063383e.svg?invert_in_darkmode" align=middle width=180.84165pt height=50.226165pt/></p>

   В случае гиперболического тангеса:

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/a2d1f5c4c1cfd4eb7b679350d8aeb85d.svg?invert_in_darkmode" align=middle width=160.38198pt height=36.27789pt/></p>

   Тогда 

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/e047b5019a674502886d7230d6dfb0ad.svg?invert_in_darkmode" align=middle width=87.65657999999999pt height=38.51529pt/></p>

<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/abe9d2643906994afdf0355a0f9f34d3.svg?invert_in_darkmode" align=middle width=86.17636499999999pt height=36.27789pt/></p>
