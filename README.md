# bmstu_dod
bmstu presentation 
Карта глубины

## Features
numpy - векторные операции

onnxruntime - использование нейросети

openCV - визуализация

## install

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd bmstu_dod
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source .venv/bin/activate
     ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
5. **большие файлы моделей**
   - детектор и ключевые точки лица (и все файлы проекта):  https://disk.yandex.ru/d/Nk70-XTLDGr6hw
   - детектор : https://disk.yandex.ru/d/dMWd9pf3l7SYIA
   - ключевые точки : https://disk.yandex.ru/d/cEWBINZ8PPP0_w
     
**Run the application:**

-c - индекс камеры, 

-s - размер окна отображения (по вертикали, по горизонтали будет в 2 раза больше)

   **только карта глубины**
   ```bash
   python models_run.py -c 0 -s 500
   ```
   **карта глубины и точки лица**
   ```bash
   python models_run_2.py -c 0 -s 500
   ```





