#!/usr/bin/env python3
"""
Production RTSP to YOLO Processor - ADVANCED OBJECT TRACKING
"""

import cv2
import time
import logging
import subprocess
import numpy as np
import select
import threading
from flask import Flask, Response
from ultralytics import YOLO
import queue
import json
from datetime import datetime
import sys
from collections import OrderedDict, deque
import scipy.spatial as sp
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rtsp_yolo_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.rtsp_url = "rtsp://admin:Jaquio@192.168.15.166:554/live"
        self.model_path = "yolov8n.pt"
        
        # ФИКСИРОВАННЫЕ РАЗМЕРЫ
        self.capture_width = 1920
        self.capture_height = 1080
        
        # Размеры для обработки YOLO
        self.processing_width = 640
        self.processing_height = 360
        
        # Размеры для веб-вывода
        self.web_width = 854
        self.web_height = 480
        
        self.target_fps = 20
        self.process_every_n = 3
        self.confidence_threshold = 0.5
        self.web_host = "0.0.0.0"
        self.web_port = 8001
        self.web_quality = 60
        
        # УЛУЧШЕННЫЕ НАСТРОЙКИ ТРЕКЕРА
        self.tracker_max_age = 30  # увеличен срок жизни объекта
        self.tracker_min_hits = 3  # минимальное количество детекций для подтверждения
        self.tracker_iou_threshold = 0.4  # более строгий порог
        self.tracker_appearance_weight = 0.7  # вес внешнего вида vs движения
        self.tracker_velocity_weight = 0.3  # вес скорости

class KalmanFilter:
    """Упрощенный Kalman фильтр для трекинга объектов"""
    
    def __init__(self):
        # Состояние: [x, y, w, h, dx, dy]
        self.state = np.zeros(6)
        # Ковариационная матрица
        self.covariance = np.eye(6) * 10
        
        # Матрица перехода (предполагаем постоянную скорость)
        self.transition_matrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Матрица наблюдения (измеряем только позицию и размер)
        self.observation_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        # Шум процесса
        self.process_noise = np.eye(6) * 0.03
        # Шум измерений
        self.measurement_noise = np.eye(4) * 0.1
    
    def init(self, bbox):
        """Инициализация фильтра с bounding box"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        self.state = np.array([cx, cy, w, h, 0, 0])
        self.covariance = np.eye(6) * 10
    
    def predict(self):
        """Предсказание следующего состояния"""
        self.state = self.transition_matrix @ self.state
        self.covariance = (self.transition_matrix @ self.covariance @ 
                          self.transition_matrix.T) + self.process_noise
        return self.get_bbox()
    
    def update(self, bbox):
        """Обновление состояния на основе измерения"""
        if bbox is None:
            return
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        measurement = np.array([cx, cy, w, h])
        
        # Innovation
        y = measurement - self.observation_matrix @ self.state
        S = self.observation_matrix @ self.covariance @ self.observation_matrix.T + self.measurement_noise
        K = self.covariance @ self.observation_matrix.T @ np.linalg.inv(S)
        
        # Обновление состояния
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ self.observation_matrix) @ self.covariance
    
    def get_bbox(self):
        """Получение bounding box из состояния"""
        cx, cy, w, h, _, _ = self.state
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]

class TrackedObject:
    """Трекаемый объект с улучшенной стабильностью ID"""
    
    def __init__(self, object_id, detection, config):
        self.object_id = object_id
        self.detection = detection
        self.class_name = detection['class_name']
        self.confidence = detection['confidence']
        
        # Kalman фильтр для сглаживания и предсказания
        self.kalman = KalmanFilter()
        self.kalman.init(detection['bbox'])
        
        # История позиций для трекинга
        self.track_history = deque(maxlen=50)
        self.update_track_history()
        
        # Счетчики для подтверждения трека
        self.hit_streak = 1  # последовательные обнаружения
        self.age = 1  # возраст трека в кадрах
        self.time_since_update = 0  # время с последнего обновления
        
        # Визуальные особенности (упрощенные)
        self.appearance_features = self._extract_appearance(detection['bbox'])
        
        self.config = config
    
    def _extract_appearance(self, bbox):
        """Упрощенное извлечение визуальных особенностей"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        return np.array([w, h, aspect_ratio, area])
    
    def update_track_history(self):
        """Обновление истории позиций"""
        bbox = self.kalman.get_bbox()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.track_history.append((cx, cy))
    
    def predict(self):
        """Предсказание следующей позиции"""
        predicted_bbox = self.kalman.predict()
        self.time_since_update += 1
        self.update_track_history()
        return predicted_bbox
    
    def update(self, detection):
        """Обновление объекта новой детекцией"""
        self.detection = detection
        self.confidence = detection['confidence']
        self.kalman.update(detection['bbox'])
        self.hit_streak += 1
        self.time_since_update = 0
        self.age += 1
        self.update_track_history()
        
        # Обновление визуальных особенностей
        self.appearance_features = self._extract_appearance(detection['bbox'])
    
    def similarity_score(self, detection):
        """Оценка схожести с новой детекцией"""
        bbox1 = self.kalman.get_bbox()
        bbox2 = detection['bbox']
        
        # 1. IoU (Intersection over Union)
        iou = self._calculate_iou(bbox1, bbox2)
        
        # 2. Схожесть классов
        class_similarity = 1.0 if self.class_name == detection['class_name'] else 0.0
        
        # 3. Схожесть размера и формы
        features1 = self.appearance_features
        features2 = self._extract_appearance(bbox2)
        size_similarity = 1.0 - min(1.0, np.linalg.norm(features1 - features2) / 100)
        
        # Комбинированная оценка
        motion_similarity = iou * self.config.tracker_velocity_weight
        appearance_similarity = (class_similarity + size_similarity) / 2 * self.config.tracker_appearance_weight
        
        return motion_similarity + appearance_similarity
    
    def _calculate_iou(self, box1, box2):
        """Вычисление Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

class AdvancedObjectTracker:
    """Продвинутый трекер объектов с стабильными ID"""
    
    def __init__(self, config):
        self.config = config
        self.next_object_id = 1
        self.tracked_objects = OrderedDict()  # object_id -> TrackedObject
        self.frames_since_update = 0
    
    def update(self, detections):
        """Обновление трекера с новыми детекциями"""
        self.frames_since_update += 1
        
        # Предсказание позиций для всех существующих объектов
        for obj in self.tracked_objects.values():
            obj.predict()
        
        # Создание матрицы схожести
        if detections and self.tracked_objects:
            similarity_matrix = self._create_similarity_matrix(detections)
            matched_pairs = self._hungarian_matching(similarity_matrix)
        else:
            matched_pairs = []
        
        # Обработка совпадений
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, track_idx in matched_pairs:
            if similarity_matrix[det_idx][track_idx] > self.config.tracker_iou_threshold:
                track_id = list(self.tracked_objects.keys())[track_idx]
                detection = detections[det_idx]
                
                self.tracked_objects[track_id].update(detection)
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
        
        # Обновление неподтвержденных треков
        for track_idx, track_id in enumerate(list(self.tracked_objects.keys())):
            if track_idx not in matched_tracks:
                obj = self.tracked_objects[track_id]
                obj.time_since_update += 1
                
                # Удаление старых треков
                if obj.time_since_update > self.config.tracker_max_age:
                    del self.tracked_objects[track_id]
        
        # Создание новых треков для неподходящих детекций
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                # Только для детекций с высокой уверенностью создаем новые треки
                if detection['confidence'] > 0.6:
                    self._create_new_track(detection)
        
        # Возврат активных треков
        active_detections = []
        for obj in self.tracked_objects.values():
            if obj.time_since_update == 0 or obj.hit_streak >= self.config.tracker_min_hits:
                detection = obj.detection.copy()
                detection['object_id'] = obj.object_id
                detection['track_history'] = obj.track_history
                detection['age'] = obj.age
                detection['hit_streak'] = obj.hit_streak
                active_detections.append(detection)
        
        return active_detections
    
    def _create_similarity_matrix(self, detections):
        """Создание матрицы схожести между детекциями и треками"""
        track_ids = list(self.tracked_objects.keys())
        similarity_matrix = np.zeros((len(detections), len(track_ids)))
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                obj = self.tracked_objects[track_id]
                similarity_matrix[det_idx][track_idx] = obj.similarity_score(detection)
        
        return similarity_matrix
    
    def _hungarian_matching(self, cost_matrix):
        """Венгерский алгоритм для оптимального сопоставления"""
        # Преобразование в матрицу стоимости (1 - схожесть)
        cost_matrix = 1 - cost_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))
    
    def _create_new_track(self, detection):
        """Создание нового трека"""
        object_id = self.next_object_id
        self.tracked_objects[object_id] = TrackedObject(object_id, detection, self.config)
        self.next_object_id += 1

class RTSPYOLOProcessor:
    def __init__(self, config):
        self.config = config
        self.frame_size = config.capture_width * config.capture_height * 3
        
        # Инициализация улучшенного трекера
        self.object_tracker = AdvancedObjectTracker(config)
        
        # ЕДИНСТВЕННЫЙ буфер для веб-вывода
        self.output_buffer = queue.Queue(maxsize=1)
        
        # Отдельный буфер для обработки
        self.processing_buffer = queue.Queue(maxsize=5)
        
        self.running = False
        self.capture_frame_count = 0
        self.processed_frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # ЕДИНСТВЕННОЕ место для хранения текущего кадра
        self._current_output_frame = self._create_info_frame("Starting...")
        self._current_detections = []
        self._frame_lock = threading.Lock()

    def _create_info_frame(self, message):
        """Создание информационного кадра"""
        frame = np.zeros((self.config.web_height, self.config.web_width, 3), dtype=np.uint8)
        
        # Градиентный фон
        for i in range(self.config.web_height):
            color = int(50 + (i / self.config.web_height) * 50)
            frame[i, :] = [color, color, color]
        
        text_y = self.config.web_height // 2
        cv2.putText(frame, message, (50, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return frame

    def start_ffmpeg(self):
        """Запуск FFmpeg с ФИКСИРОВАННЫМ разрешением"""
        try:
            command = [
                'ffmpeg',
                '-i', self.config.rtsp_url,
                '-loglevel', 'quiet',
                '-an',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vcodec', 'rawvideo',
                '-r', str(self.config.target_fps),
                '-s', f"{self.config.capture_width}x{self.config.capture_height}",
                '-'
            ]
            
            logger.info(f"🎥 Запуск FFmpeg с разрешением {self.config.capture_width}x{self.config.capture_height}")
            self.pipe = subprocess.Popen(command, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       bufsize=10**8)
            logger.info("✅ FFmpeg запущен")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка FFmpeg: {e}")
            return False

    def load_yolo_model(self):
        """Загрузка модели YOLO"""
        try:
            logger.info(f"Загрузка модели YOLO: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            logger.info("✅ Модель YOLO загружена")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки YOLO: {e}")
            return False

    def resize_frame_proportional(self, frame, target_width, target_height):
        """Изменение размера кадра с сохранением пропорций"""
        h, w = frame.shape[:2]
        
        aspect_ratio = w / h
        target_ratio = target_width / target_height
        
        if aspect_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas

    def capture_frames(self):
        """Захват кадров из RTSP - ТОЛЬКО ЗАХВАТ"""
        logger.info("🎥 Запуск захвата кадров")
        
        consecutive_errors = 0
        max_errors = 5
        
        while self.running:
            try:
                ready, _, _ = select.select([self.pipe.stdout], [], [], 1.0)
                
                if ready:
                    raw_frame = self.pipe.stdout.read(self.frame_size)
                    
                    if len(raw_frame) == self.frame_size:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8)
                        frame = frame.reshape((self.config.capture_height, self.config.capture_width, 3))
                        
                        # Кладем в буфер обработки (не блокируем если полон)
                        if not self.processing_buffer.full():
                            self.processing_buffer.put(frame)
                        
                        self.capture_frame_count += 1
                        consecutive_errors = 0  # Сбрасываем счетчик ошибок
                        
                    else:
                        logger.warning(f"Неполный кадр: {len(raw_frame)}/{self.frame_size}")
                        consecutive_errors += 1
                else:
                    logger.warning("Таймаут чтения кадра")
                    consecutive_errors += 1
                
                # Перезапуск при множественных ошибках
                if consecutive_errors >= max_errors:
                    logger.error("Слишком много ошибок, перезапуск захвата...")
                    self.restart_ffmpeg()
                    consecutive_errors = 0
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Ошибка захвата: {e}")
                consecutive_errors += 1
                time.sleep(1)

    def restart_ffmpeg(self):
        """Перезапуск FFmpeg при проблемах"""
        logger.info("Перезапуск FFmpeg...")
        if hasattr(self, 'pipe'):
            try:
                self.pipe.terminate()
                self.pipe.wait(timeout=5)
            except:
                self.pipe.kill()
        time.sleep(1)
        return self.start_ffmpeg()

    def get_latest_frame(self):
        """Получение последнего кадра - БЕЗ ОЧИСТКИ БУФЕРОВ"""
        with self._frame_lock:
            return self._current_output_frame.copy(), self._current_detections.copy()

    def _get_color_by_id(self, object_id):
        """Генерация уникального цвета на основе ID"""
        # Используем хэш для стабильных цветов
        hue = (object_id * 50) % 180  # HSV hue от 0 до 180
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        return [int(c) for c in bgr_color[0][0]]

    def process_frames(self):
        """Обработка кадров с YOLO - с улучшенным трекингом"""
        logger.info("🔍 Запуск обработки YOLO с улучшенным трекингом")
        frame_counter = 0
        
        while self.running:
            try:
                # Берем кадр из буфера обработки
                frame = self.processing_buffer.get(timeout=1.0)
                frame_counter += 1
                
                # Обрабатываем каждый N-й кадр
                if frame_counter % self.config.process_every_n == 0:
                    # Подготовка кадра для YOLO
                    processing_frame = self.resize_frame_proportional(
                        frame, 
                        self.config.processing_width, 
                        self.config.processing_height
                    )
                    
                    # YOLO обработка
                    results = self.model(processing_frame, 
                                       conf=self.config.confidence_threshold,
                                       verbose=False)
                    
                    # Извлечение детекций
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                xyxy = box.xyxy[0].tolist()
                                
                                detection = {
                                    'class': cls,
                                    'confidence': conf,
                                    'bbox': xyxy,
                                    'class_name': self.model.names[cls]
                                }
                                detections.append(detection)
                                self.detection_count += 1
                    
                    # ОБНОВЛЕНИЕ УЛУЧШЕННОГО ТРЕКЕРА
                    tracked_detections = self.object_tracker.update(detections)
                    
                    # Создание кадра для веб-вывода
                    web_frame = self.resize_frame_proportional(
                        frame,
                        self.config.web_width,
                        self.config.web_height
                    )
                    
                    # Масштабирование bounding boxes
                    scale_x = self.config.web_width / self.config.processing_width
                    scale_y = self.config.web_height / self.config.processing_height
                    
                    # Отрисовка детекций с улучшенной визуализацией
                    for det in tracked_detections:
                        x1, y1, x2, y2 = det['bbox']
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y) 
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # Уникальный цвет на основе ID
                        object_id = det.get('object_id', 0)
                        color = self._get_color_by_id(object_id)
                        
                        # Рисуем bounding box
                        cv2.rectangle(web_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Подпись с улучшенной информацией
                        age = det.get('age', 1)
                        hit_streak = det.get('hit_streak', 1)
                        label = f"ID:{object_id} {det['class_name']} {det['confidence']:.2f}"
                        sub_label = f"Age:{age} Hits:{hit_streak}"
                        
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # Фон для текста
                        cv2.rectangle(web_frame, (x1, y1-text_height-25), 
                                    (x1+text_width, y1), color, -1)
                        cv2.putText(web_frame, label, (x1, y1-15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(web_frame, sub_label, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Отрисовка истории трекинга
                        if 'track_history' in det and len(det['track_history']) > 1:
                            points = []
                            for point in det['track_history']:
                                px, py = point
                                px = int(px * scale_x)
                                py = int(py * scale_y)
                                points.append((px, py))
                            
                            # Рисуем плавную линию трекинга
                            for i in range(1, len(points)):
                                thickness = max(1, int(3 * (i / len(points))))
                                cv2.line(web_frame, points[i-1], points[i], color, thickness)
                    
                    # Улучшенная статистика
                    current_time = datetime.now().strftime("%H:%M:%S")
                    elapsed = time.time() - self.start_time
                    fps = self.processed_frame_count / elapsed if elapsed > 0 else 0
                    
                    text_x = self.config.web_width - 220
                    stats_bg = np.zeros((100, 230, 3), dtype=np.uint8)
                    stats_bg[:,:] = [0, 0, 0]
                    
                    # Накладываем полупрозрачный фон для статистики
                    web_frame[10:110, text_x-10:text_x+220] = (
                        web_frame[10:110, text_x-10:text_x+220] * 0.3 + stats_bg * 0.7
                    ).astype(np.uint8)
                    
                    cv2.putText(web_frame, f"Time: {current_time}", (text_x, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(web_frame, f"FPS: {fps:.1f}", (text_x, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(web_frame, f"Objects: {len(tracked_detections)}", (text_x, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.putText(web_frame, f"Tracks: {len(self.object_tracker.tracked_objects)}", (text_x, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # ОБНОВЛЕНИЕ с блокировкой!
                    with self._frame_lock:
                        self._current_output_frame = web_frame.copy()
                        self._current_detections = tracked_detections.copy()
                    
                    self.processed_frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка обработки: {e}")
                time.sleep(0.1)

    def start_processing(self):
        """Запуск обработки"""
        if not self.start_ffmpeg():
            return False
        
        if not self.load_yolo_model():
            return False
        
        self.running = True
        
        # Инициализация начального кадра
        with self._frame_lock:
            self._current_output_frame = self._create_info_frame("Initializing...")
        
        # Запуск потоков
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        
        capture_thread.start()
        time.sleep(3)  # Даем время на запуск захвата
        process_thread.start()
        
        logger.info("✅ Все потоки запущены")
        return True

    def start_web_server(self):
        """Запуск веб-сервера с фиксированным FPS"""
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>VisionGuard RTSP - Advanced Object Tracking</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        margin: 0; 
                        padding: 0;
                        background: #000;
                        overflow: hidden;
                        font-family: Arial, sans-serif;
                    }
                    #video {
                        width: 100vw;
                        height: 100vh;
                        object-fit: contain;
                    }
                    .stats {
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        color: white;
                        background: rgba(0,0,0,0.7);
                        padding: 10px;
                        border-radius: 5px;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <img id="video" src="/video_feed">
                <div class="stats" id="stats">Loading...</div>

                <script>
                    function refreshVideo() {
                        const video = document.getElementById('video');
                        const newSrc = '/video_feed?t=' + new Date().getTime();
                        
                        if (video.src !== newSrc) {
                            video.src = newSrc;
                        }
                    }

                    // Обновляем видео каждые 5 минут для надежности
                    setInterval(refreshVideo, 300000);

                    // Автоматический рефреш при ошибках
                    document.getElementById('video').onerror = function() {
                        setTimeout(refreshVideo, 1000);
                    };

                    // Обновление статистики
                    function updateStats() {
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('stats').innerHTML = 
                                    `Objects: ${data.objects_count}<br>
                                     FPS: ${data.fps.toFixed(1)}<br>
                                     Total Tracks: ${data.total_tracks}<br>
                                     Processed: ${data.processed_frames}`;
                            })
                            .catch(() => {
                                document.getElementById('stats').innerHTML = 'Stats unavailable';
                            });
                    }

                    setInterval(updateStats, 1000);
                    updateStats();
                </script>
            </body>
            </html>
            """
        
        @app.route('/video_feed')
        def video_feed():
            def generate():
                target_fps = 10  # Фиксированный FPS для веб-потока
                frame_interval = 1.0 / target_fps
                last_frame_time = 0
                
                while True:
                    try:
                        current_time = time.time()
                        
                        # Строгое соблюдение интервала FPS
                        if current_time - last_frame_time >= frame_interval:
                            frame, detections = self.get_latest_frame()
                            
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.web_quality]
                            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
                            
                            if success:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + 
                                       encoded_image.tobytes() + b'\r\n')
                                last_frame_time = current_time
                        
                        # Стабильная задержка
                        time.sleep(0.001)
                        
                    except Exception as e:
                        logger.error(f"Ошибка в видеопотоке: {e}")
                        time.sleep(0.1)
            
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/stats')
        def stats():
            elapsed = time.time() - self.start_time
            fps = self.processed_frame_count / elapsed if elapsed > 0 else 0
            
            return {
                'objects_count': len(self._current_detections),
                'fps': fps,
                'total_tracks': len(self.object_tracker.tracked_objects),
                'processed_frames': self.processed_frame_count,
                'total_detections': self.detection_count
            }
        
        logger.info(f"🌐 Запуск веб-сервера на http://{self.config.web_host}:{self.config.web_port}")
        app.run(host=self.config.web_host, port=self.config.web_port, threaded=True, debug=False)

    def start(self):
        """Запуск всей системы"""
        if not self.start_processing():
            return False
        
        self.start_web_server()
        return True

    def stop(self):
        """Остановка"""
        self.running = False
        if hasattr(self, 'pipe'):
            self.pipe.terminate()

def main():
    config = Config()
    processor = RTSPYOLOProcessor(config)
    
    try:
        if processor.start():
            logger.info("✅ Система запущена с улучшенным трекингом")
        else:
            logger.error("❌ Не удалось запустить систему")
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()