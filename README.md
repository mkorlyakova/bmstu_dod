# bmstu_dod
bmstu presentation 
Карта глубины

## Features
numpy - векторные операции
onnxruntime - использование нейросети
openCV - визуализация

# install

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

**Run the application:**
-c - индекс камеры
-s - размер окна отображения (по вертикали, по горизонтали будет в 2 раза больше)

   ```bash
   python models_run.py -c 0 -s 500
   ```





