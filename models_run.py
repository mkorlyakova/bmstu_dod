#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


parser = argparse.ArgumentParser(description="Карта глубины")

# Добавляем аргументы
parser.add_argument("-s", "--size", type=int, default=500, help="Введите размер кадра")
parser.add_argument("-c", "--cam", type=str, default="0", help="Введите нопер камеры")
args = parser.parse_args()

# ==========================================
#              КОНФИГУРАЦИЯ
# ==========================================

SIZE = args.size
CAM = args.cam

# --- РЕЖИМ РАБОТЫ ---
# 'video' - для камеры или видеофайла (.mp4, .avi)
MODE = "video"

# --- ВХОД / ВЫХОД ---
# Для камеры  INPUT_PATH = 0 (числом, без кавычек) или строку пути к камере
INPUT_PATH = int(CAM) if CAM.isdigit() else CAM

OUTPUT_PATH = "test_output.avi"

# --- МОДЕЛИ ---
YOLO_MODEL = "./trained_models/detect128.onnx"
DEPTH_MODEL = "./mv2_linknet_128.onnx"

# --- РАЗМЕРЫ ВХОДА ---
# YOLO_SIZE = 128
DEPTH_SIZE = 128

# --- ОПЦИИ ---
SHOW_DISPLAY = True  # Показывать окно
ENABLE_RECORDING = False  # Сохранять результат в файл
ENABLE_YOLO = False  # Детекция
ENABLE_DEPTH = True  # Глубина

# --- НАСТРОЙКИ КАМЕРЫ (Только для режима video/camera) ---
CAM_WIDTH = 160
CAM_HEIGHT = 120
SKIP_FRAMES = 2

# ==========================================

print("--- CONFIG ---")
print("Mode:   {}".format(MODE))
print("Input:  {}".format(INPUT_PATH))
print("Models: YOLO={}, Depth={}".format(ENABLE_YOLO, ENABLE_DEPTH))

# --- Инициализация ONNX ---
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess_yolo = None
sess_depth = None
yolo_in, yolo_out = None, None
depth_in = None

try:

    if ENABLE_DEPTH:
        print("Loading Depth...")
        sess_depth = ort.InferenceSession(
            DEPTH_MODEL, sess_options, providers=["CPUExecutionProvider"]
        )
        depth_in = sess_depth.get_inputs()[0].name

except Exception as e:
    print("Error loading models: {}".format(e))
    exit()

# Массивы для нормализации (Depth)
mean_val = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std_val = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_inference(frame, w, h):
    final_boxes = []
    d_vis = np.zeros((h, w, 3), dtype=np.uint8)

    if ENABLE_DEPTH:
        small_d = cv2.resize(frame, (DEPTH_SIZE, DEPTH_SIZE))
        img_d = cv2.cvtColor(small_d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        input_depth = (img_d - mean_val) / std_val
        input_depth = np.transpose(input_depth, (2, 0, 1))
        input_depth = np.expand_dims(input_depth, axis=0)

        res_d = sess_depth.run(None, {depth_in: input_depth})[0]

        d_map = res_d[0, 0] if res_d.shape[1] == 1 else res_d[0]
        d_resized = cv2.resize(d_map, (w, h), interpolation=cv2.INTER_NEAREST)
        d_norm = cv2.normalize(d_resized, None, 0, 255, cv2.NORM_MINMAX)
        d_vis = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

    return final_boxes, d_vis


# ==========================================
#           ОСНОВНОЙ ЦИКЛ
# ==========================================


if MODE == "video":
    # --- ОБРАБОТКА ВИДЕО ---
    print("Processing Video/Camera...")

    # cap = cv2.VideoCapture(INPUT_PATH)
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Error: Cannot open video input")
        exit()

    # Настройки камеры (работают только для USB-камеры)
    cv2.namedWindow("depth_map", cv2.WINDOW_NORMAL)
    if INPUT_PATH == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resolution: {}x{}".format(w, h))

    # Настройка записи
    out = None
    if ENABLE_RECORDING:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # Для видео fps ставим 15 или 30
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 15.0, (w * 2, h))
        print("Recording to {}".format(OUTPUT_PATH))

    # Кеш
    cached_boxes = []
    cached_depth_vis = np.zeros((h, w, 3), dtype=np.uint8)

    frame_cnt = 0
    start_time = time.time()

    if SHOW_DISPLAY:
        print("Press 'q' to exit.")
    else:
        print("Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_cnt += 1

            # Инференс с пропуском
            if (frame_cnt - 1) % (SKIP_FRAMES + 1) == 0:
                cached_boxes, cached_depth_vis = run_inference(frame, w, h)

            # FPS
            elapsed = time.time() - start_time
            fps = frame_cnt / elapsed if elapsed > 0 else 0

            # Отрисовка
            if SHOW_DISPLAY or ENABLE_RECORDING:
                draw_img = frame.copy()
                if ENABLE_YOLO:
                    for i in range(len(cached_boxes)):
                        bx, by, bw, bh = cached_boxes[i]

                        # --- РАСЧЕТ ЦЕНТРА ---
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)

                        # Рисуем прямоугольник
                        cv2.rectangle(
                            draw_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2
                        )
                        # Рисуем центр (красная точка)
                        cv2.circle(draw_img, (cx, cy), 4, (0, 0, 255), -1)
                        # Пишем координаты рядом с объектом
                        cv2.putText(
                            draw_img,
                            "({}, {})".format(cx, cy),
                            (bx, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

                combined = np.hstack((draw_img, cached_depth_vis))
                cv2.putText(
                    combined,
                    "FPS: {:.1f}".format(fps),
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                if ENABLE_RECORDING and out is not None:
                    out.write(combined)

                if SHOW_DISPLAY:
                    cv2.resizeWindow("depth_map", SIZE * 2, SIZE)
                    cv2.imshow("depth_map", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_cnt % 30 == 0:
                # В консоль тоже выводим координаты первого найденного объекта (для отладки)
                center_info = ""
                if len(cached_boxes) > 0:
                    bx, by, bw, bh = cached_boxes[0]
                    cx = int(bx + bw / 2)
                    cy = int(by + bh / 2)
                    center_info = "| Center: ({}, {})".format(cx, cy)

                print(
                    "Frame: {} | Avg FPS: {:.2f} {}".format(frame_cnt, fps, center_info)
                )

    except KeyboardInterrupt:
        print("\nStopped.")

    cap.release()
    if out is not None:
        out.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()

else:
    print("Error: Unknown MODE '{}'".format(MODE))
