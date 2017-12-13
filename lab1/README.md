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
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/b36cdd66cc930db82e0861639504c0eb.svg?invert_in_darkmode" align=middle width=413.50154999999995pt height=50.226165pt/></p>

Функция активации на скрытом слое тангес:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/ed95e6fc19764479cbcac06b6293beab.svg?invert_in_darkmode" align=middle width=117.602925pt height=37.147275pt/></p>

Функция активации на втором слое softmax:
<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/f8ce48e4708e476899413ca68cba9971.svg?invert_in_darkmode" align=middle width=126.424485pt height=40.62036pt/></p>

### Алгоритм
1. Инициализируем веса значениями из диапазона [0,0.5]
2. Пока количество проходов < max_epoch делаем:

	Для всех картинок от 1 до number_train_images
	+ Подаем на вход x, суммируем cигналы на скрытом слое <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/24e5debb20cdd9d2d9a3364b5f0e09b4.svg?invert_in_darkmode" align=middle width=147.64645499999997pt height=47.988764999999994pt/></p>, применяем функцию активации <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/7912182562a715a981fff4ab40bfb87a.svg?invert_in_darkmode" align=middle width=74.16288pt height=16.438356pt/></p>

	+ Для каждого выходного нейрона суммируем взвешенные входящие сигналы <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/cacf919e976e1bdd2a2e0f6b2874fbfa.svg?invert_in_darkmode" align=middle width=172.5438pt height=47.49789pt/></p>, применяем функцию1 активации <p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/3fe630f6471f966a0ee2f8751b8ffbfc.svg?invert_in_darkmode" align=middle width=75.84291pt height=17.031959999999998pt/></p>

	+ Считаем градиенты функции ошибки:

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/33a5033e2a26ab9a0fdd1e5448f360f8.svg?invert_in_darkmode" align=middle width=125.15744999999998pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/f31d0c7bd21cc5253aa7b195bbef0523.svg?invert_in_darkmode" align=middle width=96.984855pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/73d12a65bbf307b427bcf40f4e138867.svg?invert_in_darkmode" align=middle width=73.42417499999999pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/902c32840fc99fdf2a8963a1f8f77d64.svg?invert_in_darkmode" align=middle width=186.64469999999997pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/c8a6c938ccd9641d27cb2404ac4cf16c.svg?invert_in_darkmode" align=middle width=121.93533pt height=36.27789pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/e5fe7703c9325ead4c97db212f417354.svg?invert_in_darkmode" align=middle width=427.55789999999996pt height=50.226165pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/cb94e20f4bd01e87c2e1c95ef063383e.svg?invert_in_darkmode" align=middle width=180.84165pt height=50.226165pt/></p>

 	 В случае гиперболического тангеса: 

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/a2d1f5c4c1cfd4eb7b679350d8aeb85d.svg?invert_in_darkmode" align=middle width=160.38198pt height=36.27789pt/></p>

 	 Тогда 

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/e047b5019a674502886d7230d6dfb0ad.svg?invert_in_darkmode" align=middle width=87.65657999999999pt height=38.51529pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/abe9d2643906994afdf0355a0f9f34d3.svg?invert_in_darkmode" align=middle width=86.17636499999999pt height=36.27789pt/></p>
	+ Корректируем веса в соответствии с градиентами функции ошибки:

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/e3b96ae350a7aaac265e6ce1ad594b5a.svg?invert_in_darkmode" align=middle width=155.663805pt height=36.27789pt/></p>

	<p align="center"><img src="https://rawgit.com/ViktorRoy94/deep_learning/master//lab1/tex/b1af6355d19d273f5321c89b4e081f8d.svg?invert_in_darkmode" align=middle width=160.22423999999998pt height=38.51529pt/></p>
