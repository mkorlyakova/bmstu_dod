#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# from PIL import Image

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
# INPUT_PATH = "/home/mariya/PycharmProjects/DOD2026/bmstu_dod/nn_11output_video1693469566.9097092.mp4"
OUTPUT_PATH = "test_output.avi"

# --- МОДЕЛИ ---
YOLO_MODEL = "./trained_models/detect128.onnx"
DEPTH_MODEL = "./mv2_linknet_128.onnx"
PATH_TO_FACEDETECTOR = "./facedetector_v2_20241007_2200.onnx"
PATH_TO_LANDMARKDETECTOR = "./landmarkdetector_v2_20241007_2200.onnx"

# --- РАЗМЕРЫ ВХОДА ---
# YOLO_SIZE = 128
DEPTH_SIZE = 128  # 128

# --- ОПЦИИ ---
SHOW_DISPLAY = True  # Показывать окно
ENABLE_RECORDING = False  # Сохранять результат в файл
ENABLE_YOLO = False  # Детекция
ENABLE_DEPTH = True  # Глубина
ENABLE_POINT = True  # False  # точки

# --- НАСТРОЙКИ КАМЕРЫ (Только для режима video/camera) ---
CAM_WIDTH = 320
CAM_HEIGHT = 240
SKIP_FRAMES = 2

# ==========================================

TRESHOLDS = 0.8
LANDMARK_SIZE = 192
DETECTOR_SIZE = 640

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
    if ENABLE_POINT:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        #
        try:
            sess_landmark = ort.InferenceSession(
                PATH_TO_LANDMARKDETECTOR,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            sess_detector = ort.InferenceSession(
                PATH_TO_FACEDETECTOR, sess_options, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            ENABLE_POINT = False
            print("error: ", str(e))


except Exception as e:
    print("Error loading models: {}".format(e))
    exit()


def crop_faces(image_full, w_f, h_f) -> np.ndarray:
    outputs2 = [
        "fc1",
    ]
    outputs1 = [
        "scores",
        "bboxes",
        "kpss",
    ]

    im_to_detect = cv2.resize(
        image_full, dsize=(DETECTOR_SIZE, DETECTOR_SIZE), interpolation=cv2.INTER_CUBIC
    )
    size = LANDMARK_SIZE
    scores, bboxes, kpss, lmks = [], [], [], []
    out = sess_detector.run(outputs1, {"input": np.array(im_to_detect)})

    scores, bboxes, kpss = out[0], out[1], out[2]

    ind = np.argsort(scores)

    lmks = []
    imgs = []
    scores_end = []
    bboxes_end = []
    lmks5 = []

    for i in ind[-1:]:
        if scores[i] > TRESHOLDS / 2:
            scores_end.append(scores[i])

            w, h = bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]
            wh = max(w, h)
            dx = (wh - w) // 2
            dy = (wh - h) // 2
            bbox = [
                bboxes[i][0] - dx,
                bboxes[i][1] - dy,
                bboxes[i][2] + dx,
                bboxes[i][3] + dy,
            ]
            bbox = [max([b, 0]) for b in bbox]
            print(bboxes[i])
            print(bbox)
            # bboxes_end.append(bbox)
            start_point = [
                (float(bbox[0]) * 1.0 / DETECTOR_SIZE),
                (float(bbox[2]) * 1.0 / DETECTOR_SIZE),
            ]
            end_point = [
                (float(bbox[1]) * 1.0 / DETECTOR_SIZE),
                (float(bbox[3]) * 1.0 / DETECTOR_SIZE),
            ]
            bboxes_end.append(start_point + end_point)
            # cv2.rectangle(image_full, start_point, end_point, "red", 5)

            im_b = im_to_detect[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            # print(im_b.shape)
            image_fulli = cv2.resize(
                im_b,
                dsize=(size, size),
                interpolation=cv2.INTER_CUBIC,
            )

            # image_fulli_d = image_fulli.resize([size, size])

            out2 = sess_landmark.run(
                outputs2,
                {
                    "data": np.array(image_fulli)
                    .astype(np.float32)
                    .transpose(2, 0, 1)[None]
                },
            )

            imgs.append(image_fulli)
            landmarks1 = np.array(out2[0]).reshape([-1, 2]) * np.array([[wh, wh]])
            landmarks1[:, 0] += bboxes[i][0] - dx
            landmarks1[:, 1] += bboxes[i][1] - dy
            lmks.append(landmarks1)

            lmks5.append([landmarks1[[38, 88, 86, 52, 61], :]])

    if len(scores_end) == 0:
        return None, None, None, None, None
    lmks5 = np.array(lmks5).reshape([-1, 5, 2])
    lmks = np.array(lmks).reshape([-1, 106, 2])
    lmks[:, :, 0] = lmks[:, :, 0] / DETECTOR_SIZE
    lmks[:, :, 1] = lmks[:, :, 1] / DETECTOR_SIZE
    bboxes_end = np.array(bboxes_end).reshape([-1, 4])
    return (
        # image_full,
        scores_end,
        bboxes_end,  # .astype(np.int32),
        lmks5.astype(np.int32),
        # imgs,
        lmks,  # .astype(np.int32),
    )


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
                _, cached_depth_vis = run_inference(frame, w, h)

            # FPS
            elapsed = time.time() - start_time
            fps = frame_cnt / elapsed if elapsed > 0 else 0

            # Отрисовка
            if SHOW_DISPLAY or ENABLE_RECORDING:
                draw_img = frame.copy()
                # print(draw_img.shape)

                if ENABLE_POINT:
                    rez = crop_faces(frame, w, h)
                    # print(rez)
                    if rez[1] is not None:
                        print("================")
                        # print(rez[1])
                        for i in range(len(rez[1])):
                            print(rez[1][i])
                            print("--------------")
                            bx1, bx2, by1, by2 = rez[1][i]
                            bw = bx2 - bx1
                            bh = by2 - by1
                            # --- РАСЧЕТ ЦЕНТРА ---
                            cx = int((bx1 + (bx2 - bx1) / 2) * w)
                            cy = int((by1 + (by2 - by1) / 2) * h)

                            # Рисуем прямоугольник
                            print(
                                (int(bx1 * w), int(by1 * h)),
                                (int(bx2 * w), int(by2 * h)),
                            )
                            cv2.rectangle(
                                draw_img,
                                (int(bx1 * w), int(by1 * h)),
                                (int(bx2 * w), int(by2 * h)),
                                (0, 0, 255),
                                2,
                            )
                            # Рисуем центр (красная точка)
                            cv2.circle(draw_img, (cx, cy), 4, (0, 0, 255), -1)
                            # Пишем координаты рядом с объектом
                            cv2.putText(
                                draw_img,
                                "({}, {})".format(cx, cy),
                                (int(bx1 * w), int(by1 * h) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                1,
                            )
                    if rez[-1] is not None:
                        for k in range(rez[-1].shape[0]):
                            for m in range(rez[-1].shape[1]):
                                cx = int(rez[-1][k, m, 0] * w)
                                cy = int(rez[-1][k, m, 1] * h)
                                cv2.circle(
                                    cached_depth_vis, (cx, cy), 1, (255, 255, 255), -1
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
