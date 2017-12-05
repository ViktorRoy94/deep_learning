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

Функция активации softmax:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/f8ce48e4708e476899413ca68cba9971.svg?invert_in_darkmode" align=middle width=126.424485pt height=40.62036pt/></p>

### Алгоритм
1. Инициализируем веса <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/2cf68af2a7f22008cd34c07d0a454c09.svg?invert_in_darkmode" align=middle width=23.445674999999998pt height=9.5433525pt/></p> и <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/2de931a49231aca051b8756cffd491b7.svg?invert_in_darkmode" align=middle width=24.899325pt height=11.780802pt/></p> значениями из [0,0.5]
2. Пока количество проходов < max_epoch делаем:

	Для всех картинок от 1 до number_train_images
	1. Подаем на вход <img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/b9eca1ddce4792c16c191b401200a61f.svg?invert_in_darkmode" align=middle width=4.650904499999999pt height=7.614155999999999pt/></p> и суммируем cигналы на скрытом слое <img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/7a988675c564ffac06150dd992021e4b.svg?invert_in_darkmode" align=middle width=140.00184pt height=47.988764999999994pt/></p> и применяем функцию активации <img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/be8d97f702b2072622e00e6a24a95dbd.svg?invert_in_darkmode" align=middle width=66.217635pt height=16.438356pt/></p>
    2. Для каждого выходного нейрона суммируем взвешенные входящие сигналы  $$\inliney_j=w_0_j+\sum_{s}^{N_s}{w_s_j}f({z_s})$$ и применяем функцию активации $$\inlineu_j=f(y_j)$$
	3. Считаем градиенты функции ошибки:
        
