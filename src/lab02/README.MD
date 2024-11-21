1) Построение и эквализация гистограммы изображения + линейное контрастирование

5) Реализация нелинейных фильтров, основанных на порядковых статистиках



Компиляция кода:

g++ task_1.cpp -o rasterization `pkg-config --cflags --libs opencv4`
g++ task_5.cpp -o rasterization `pkg-config --cflags --libs opencv4`

Запуск программы:

./task_1
./task_5