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
1. Инициализируем веса <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/a0d6ea3432abc69fc2a52ed57a790aa3.svg?invert_in_darkmode" align=middle width=23.445674999999998pt height=9.5433525pt/></p> и <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/052c8428e2bc2be713c980ff0ff6ac6f.svg?invert_in_darkmode" align=middle width=24.899325pt height=11.780802pt/></p>
2. Пока количество проходов < max_epoch делаем:
	Для всех картинок от 1 до number_train_images
	1. Подаем на вход <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/96de47a534893e2f93c9edceffaef3d1.svg?invert_in_darkmode" align=middle width=14.045888999999999pt height=9.5433525pt/></p> и рассчитываем выходы <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/3249dfc9be831b7a39b4b4fb06d83d8c.svg?invert_in_darkmode" align=middle width=14.1639465pt height=11.780802pt/></p>
	2. Считаем градиенты функции ошибки:

	3. <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning_lab1/master//tex/822cfe57ed033682517e44a30a0cf134.svg?invert_in_darkmode" align=middle width=23.34585pt height=11.780802pt/></p>



