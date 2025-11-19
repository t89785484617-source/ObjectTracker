#!/usr/bin/env python3
"""
Parking Lot Car Counter - RTSP to YOLO with Slanted Line Crossing Detection
"""

import cv2
import time
import logging
import subprocess
import numpy as np
import select
import threading
from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
import queue
import json
from datetime import datetime
import sys
from collections import OrderedDict, deque
import scipy.spatial as sp
from scipy.optimize import linear_sum_assignment

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parking_counter.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ParkingConfig:
    def __init__(self):
        # RTSP источник
        self.rtsp_url = "rtsp://admin:Jaquio@172.30.0.68:554/live/main"
        self.model_path = "yolov8n.pt"
        
        # Размеры обработки
        self.capture_width = 1920
        self.capture_height = 1080
        self.processing_width = 640
        self.processing_height = 360
        self.web_width = 854
        self.web_height = 480
        
        # Настройки обработки
        self.target_fps = 20
        self.process_every_n = 2
        self.confidence_threshold = 0.5
        
        # Веб-интерфейс
        self.web_host = "0.0.0.0"
        self.web_port = 8001
        self.web_quality = 60
        
        # НАСТРОЙКИ ДЛЯ ПАРКОВКИ
        self.car_classes = [2, 5, 7]  # car, bus, truck в COCO
        self.tracker_max_age = 45
        self.tracker_min_hits = 3
        self.tracker_iou_threshold = 0.3
        
        # НАКЛОННАЯ ЛИНИЯ ПОДСЧЕТА (настройте под вашу камеру)
        # Формат: [(x1, y1), (x2, y2)] в относительных координатах (0-1)
        self.counting_line = [(0.0, 0.8), (0.7, 0.3)]  # Ваша настройка
        self.counting_direction = "up"  # "up" или "down"

class KalmanFilter:
    """Упрощенный Kalman фильтр для трекинга автомобилей"""
    
    def __init__(self):
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 10
        
        self.transition_matrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        self.observation_matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        self.process_noise = np.eye(6) * 0.03
        self.measurement_noise = np.eye(4) * 0.1
    
    def init(self, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        self.state = np.array([cx, cy, w, h, 0, 0])
        self.covariance = np.eye(6) * 10
    
    def predict(self):
        self.state = self.transition_matrix @ self.state
        self.covariance = (self.transition_matrix @ self.covariance @ 
                          self.transition_matrix.T) + self.process_noise
        return self.get_bbox()
    
    def update(self, bbox):
        if bbox is None:
            return
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        measurement = np.array([cx, cy, w, h])
        
        y = measurement - self.observation_matrix @ self.state
        S = self.observation_matrix @ self.covariance @ self.observation_matrix.T + self.measurement_noise
        K = self.covariance @ self.observation_matrix.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ self.observation_matrix) @ self.covariance
    
    def get_bbox(self):
        cx, cy, w, h, _, _ = self.state
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return [x1, y1, x2, y2]

class TrackedVehicle:
    """Трекаемый автомобиль с подсчетом пересечений наклонной линии"""
    
    def __init__(self, object_id, detection, config):
        self.object_id = object_id
        self.detection = detection
        self.class_name = detection['class_name']
        self.confidence = detection['confidence']
        
        # Kalman фильтр
        self.kalman = KalmanFilter()
        self.kalman.init(detection['bbox'])
        
        # История позиций
        self.track_history = deque(maxlen=30)
        
        # Статус трекинга
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        
        # Для подсчета пересечений линии
        self.last_position = None
        self.has_crossed_line = False
        self.crossing_direction = None
        self.last_side = None  # На какой стороне линии находился в последний раз
        
        self.config = config
        self.update_track_history()
    
    def update_track_history(self):
        bbox = self.kalman.get_bbox()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.track_history.append((cx, cy))
        
        current_pos = cy
        if self.last_position is not None:
            if current_pos < self.last_position:
                self.crossing_direction = "up"
            else:
                self.crossing_direction = "down"
        self.last_position = current_pos
    
    def predict(self):
        predicted_bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        self.update_track_history()
        return predicted_bbox
    
    def update(self, detection):
        self.detection = detection
        self.confidence = detection['confidence']
        self.kalman.update(detection['bbox'])
        self.hit_streak += 1
        self.time_since_update = 0
        self.update_track_history()
    
    def _point_side_of_line(self, point, line_start, line_end):
        """Определяет на какой стороне линии находится точка"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Векторное произведение для определения стороны
        d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        return "left" if d > 0 else "right"
    
    def check_line_crossing(self, line_start, line_end):
        """Проверка пересечения наклонной линии подсчета - УЛУЧШЕННАЯ ВЕРСИЯ"""
        if len(self.track_history) < 2:
            return False, None
        
        current_point = self.track_history[-1]
        previous_point = self.track_history[-2]
        
        # Определяем на какой стороне линии находятся точки
        current_side = self._point_side_of_line(current_point, line_start, line_end)
        previous_side = self._point_side_of_line(previous_point, line_start, line_end)
        
        # Если стороны разные - произошло пересечение
        if current_side != previous_side and not self.has_crossed_line:
            # Определяем направление относительно конфига
            if self.config.counting_direction == "up":
                # Для направления "up" считаем переход справа налево как въезд
                direction = "up" if current_side == "left" and previous_side == "right" else "down"
            else:
                # Для направления "down" считаем переход слева направо как выезд  
                direction = "down" if current_side == "right" and previous_side == "left" else "up"
            
            self.has_crossed_line = True
            self.crossing_direction = direction
            logger.info(f"🚗 Пересечение линии! ID:{self.object_id} Направление: {direction}")
            return True, direction
        
        return False, None
    
    def similarity_score(self, detection):
        """Оценка схожести с новой детекцией"""
        bbox1 = self.kalman.get_bbox()
        bbox2 = detection['bbox']
        
        iou = self._calculate_iou(bbox1, bbox2)
        class_similarity = 1.0 if self.class_name == detection['class_name'] else 0.0
        
        return iou * 0.7 + class_similarity * 0.3
    
    def _calculate_iou(self, box1, box2):
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

class ParkingLotTracker:
    """Трекер для парковки с подсчетом автомобилей через наклонную линию"""
    
    def __init__(self, config):
        self.config = config
        self.next_object_id = 1
        self.tracked_vehicles = OrderedDict()
        
        # Статистика парковки
        self.vehicles_in = 0
        self.vehicles_out = 0
        self.current_vehicles = 0
        self.initial_count = 0  # Начальное количество машин
        
    def set_initial_count(self, count):
        """Установить начальное количество машин на парковке"""
        self.initial_count = count
        self.current_vehicles = count
        logger.info(f"🎯 Установлено начальное количество машин: {count}")
        
    def update(self, detections):
        """Обновление трекера с подсчетом пересечений"""
        
        # Предсказание позиций
        for vehicle in self.tracked_vehicles.values():
            vehicle.predict()
        
        # Сопоставление детекций с существующими треками
        if detections and self.tracked_vehicles:
            similarity_matrix = self._create_similarity_matrix(detections)
            matched_pairs = self._hungarian_matching(similarity_matrix)
        else:
            matched_pairs = []
        
        # Обработка совпадений
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, track_idx in matched_pairs:
            if similarity_matrix[det_idx][track_idx] > self.config.tracker_iou_threshold:
                track_id = list(self.tracked_vehicles.keys())[track_idx]
                detection = detections[det_idx]
                
                self.tracked_vehicles[track_id].update(detection)
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
        
        # Удаление старых треков
        for track_idx, track_id in enumerate(list(self.tracked_vehicles.keys())):
            if track_idx not in matched_tracks:
                vehicle = self.tracked_vehicles[track_id]
                vehicle.time_since_update += 1
                
                if vehicle.time_since_update > self.config.tracker_max_age:
                    del self.tracked_vehicles[track_id]
        
        # Создание новых треков
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                if detection['confidence'] > 0.6:
                    self._create_new_track(detection)
        
        # Проверка пересечений линии и обновление статистики
        self._check_line_crossings()
        
        # Обновление текущего количества автомобилей
        # current_vehicles = initial_count + (въехало - выехало)
        self.current_vehicles = max(0, self.initial_count + (self.vehicles_in - self.vehicles_out))
        
        # Возврат активных треков
        active_detections = []
        for vehicle in self.tracked_vehicles.values():
            if vehicle.time_since_update == 0 or vehicle.hit_streak >= self.config.tracker_min_hits:
                detection = vehicle.detection.copy()
                detection['object_id'] = vehicle.object_id
                detection['track_history'] = vehicle.track_history
                detection['has_crossed_line'] = vehicle.has_crossed_line
                active_detections.append(detection)
        
        return active_detections
    
    def _check_line_crossings(self):
        """Проверка пересечений наклонной линии подсчета для всех автомобилей"""
        # Конвертируем относительные координаты линии в абсолютные для processing кадра
        line_start = (
            self.config.counting_line[0][0] * self.config.processing_width,
            self.config.counting_line[0][1] * self.config.processing_height
        )
        line_end = (
            self.config.counting_line[1][0] * self.config.processing_width,
            self.config.counting_line[1][1] * self.config.processing_height
        )
        
        for vehicle in self.tracked_vehicles.values():
            crossed, direction = vehicle.check_line_crossing(line_start, line_end)
            
            if crossed:
                if direction == "up":
                    self.vehicles_in += 1
                    logger.info(f"🚗 ВЪЕХАЛА машина! ID:{vehicle.object_id} Всего въехало: {self.vehicles_in}")
                else:
                    self.vehicles_out += 1
                    logger.info(f"🚗 ВЫЕХАЛА машина! ID:{vehicle.object_id} Всего выехало: {self.vehicles_out}")
    
    def _create_similarity_matrix(self, detections):
        track_ids = list(self.tracked_vehicles.keys())
        similarity_matrix = np.zeros((len(detections), len(track_ids)))
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                vehicle = self.tracked_vehicles[track_id]
                similarity_matrix[det_idx][track_idx] = vehicle.similarity_score(detection)
        
        return similarity_matrix
    
    def _hungarian_matching(self, cost_matrix):
        cost_matrix = 1 - cost_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))
    
    def _create_new_track(self, detection):
        object_id = self.next_object_id
        self.tracked_vehicles[object_id] = TrackedVehicle(object_id, detection, self.config)
        self.next_object_id += 1
        logger.info(f"🆕 Новый трек: ID:{object_id} {detection['class_name']}")

class ParkingLotProcessor:
    def __init__(self, config):
        self.config = config
        self.frame_size = config.capture_width * config.capture_height * 3
        
        # Инициализация трекера парковки
        self.parking_tracker = ParkingLotTracker(config)
        
        # Буферы
        self.processing_buffer = queue.Queue(maxsize=5)
        
        self.running = False
        self.capture_frame_count = 0
        self.processed_frame_count = 0
        self.start_time = time.time()
        
        # Текущий кадр для вывода
        self._current_output_frame = self._create_info_frame("Starting Parking Lot Monitor...")
        self._current_detections = []
        self._frame_lock = threading.Lock()

    def _create_info_frame(self, message):
        frame = np.zeros((self.config.web_height, self.config.web_width, 3), dtype=np.uint8)
        
        for i in range(self.config.web_height):
            color = int(50 + (i / self.config.web_height) * 50)
            frame[i, :] = [color, color, color]
        
        text_y = self.config.web_height // 2
        cv2.putText(frame, message, (50, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return frame

    def start_ffmpeg(self):
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
            
            logger.info(f"🎥 Запуск FFmpeg для парковки")
            self.pipe = subprocess.Popen(command, 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       bufsize=10**8)
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка FFmpeg: {e}")
            return False

    def load_yolo_model(self):
        try:
            logger.info(f"Загрузка модели YOLO: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            logger.info("✅ Модель YOLO загружена")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки YOLO: {e}")
            return False

    def resize_frame_proportional(self, frame, target_width, target_height):
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
        logger.info("🎥 Запуск захвата кадров для парковки")
        
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
                        
                        if not self.processing_buffer.full():
                            self.processing_buffer.put(frame)
                        
                        self.capture_frame_count += 1
                        consecutive_errors = 0
                    else:
                        logger.warning(f"Неполный кадр: {len(raw_frame)}/{self.frame_size}")
                        consecutive_errors += 1
                else:
                    consecutive_errors += 1
                
                if consecutive_errors >= max_errors:
                    logger.error("Перезапуск захвата...")
                    self.restart_ffmpeg()
                    consecutive_errors = 0
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Ошибка захвата: {e}")
                consecutive_errors += 1
                time.sleep(1)

    def restart_ffmpeg(self):
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
        with self._frame_lock:
            return self._current_output_frame.copy(), self._current_detections.copy()

    def _get_color_by_id(self, object_id):
        hue = (object_id * 50) % 180
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        return [int(c) for c in bgr_color[0][0]]

    def _draw_parking_info(self, frame):
        """Отрисовка информации о парковке на кадре"""
        h, w = frame.shape[:2]
        
        # Наклонная линия подсчета (оставляем только линию, без текста)
        line_start = (
            int(self.config.counting_line[0][0] * w),
            int(self.config.counting_line[0][1] * h)
        )
        line_end = (
            int(self.config.counting_line[1][0] * w),
            int(self.config.counting_line[1][1] * h)
        )
        
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
        
        # Статистика парковки
        stats_bg = np.zeros((120, 300, 3), dtype=np.uint8)
        stats_bg[:] = (0, 0, 0)
        
        # Позиционируем статистику в левом верхнем углу
        x_offset, y_offset = 10, 10
        
        cv2.putText(frame, f"IN: {self.parking_tracker.vehicles_in}", 
                   (x_offset, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.parking_tracker.vehicles_out}", 
                   (x_offset, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"NOW: {self.parking_tracker.current_vehicles}", 
                   (x_offset, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"INITIAL: {self.parking_tracker.initial_count}", 
                   (x_offset, y_offset + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return frame

    def process_frames(self):
        logger.info("🔍 Запуск обработки для парковки")
        frame_counter = 0
        
        while self.running:
            try:
                frame = self.processing_buffer.get(timeout=1.0)
                frame_counter += 1
                
                if frame_counter % self.config.process_every_n == 0:
                    # Подготовка кадра для YOLO
                    processing_frame = self.resize_frame_proportional(
                        frame, 
                        self.config.processing_width, 
                        self.config.processing_height
                    )
                    
                    # YOLO обработка ТОЛЬКО автомобилей
                    results = self.model(processing_frame, 
                                       conf=self.config.confidence_threshold,
                                       classes=self.config.car_classes,
                                       verbose=False)
                    
                    # Извлечение детекций автомобилей
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
                    
                    # Обновление трекера парковки
                    try:
                        tracked_detections = self.parking_tracker.update(detections)
                    except Exception as e:
                        logger.error(f"Ошибка трекинга: {e}")
                        tracked_detections = []
                    
                    # Создание кадра для веб-вывода
                    web_frame = self.resize_frame_proportional(
                        frame,
                        self.config.web_width,
                        self.config.web_height
                    )
                    
                    # Масштабирование bounding boxes
                    scale_x = self.config.web_width / self.config.processing_width
                    scale_y = self.config.web_height / self.config.processing_height
                    
                    # Отрисовка детекций
                    for det in tracked_detections:
                        try:
                            x1, y1, x2, y2 = det['bbox']
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y) 
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            object_id = det.get('object_id', 0)
                            color = self._get_color_by_id(object_id)
                            
                            # Рисуем bounding box
                            cv2.rectangle(web_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Подпись
                            label = f"ID:{object_id} {det['class_name']} {det['confidence']:.2f}"
                            if det.get('has_crossed_line', False):
                                label += " COUNTED"
                            
                            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(web_frame, (x1, y1-text_height-10), 
                                        (x1+text_width, y1), color, -1)
                            cv2.putText(web_frame, label, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Отрисовка истории трекинга
                            if 'track_history' in det and len(det['track_history']) > 1:
                                points = []
                                for point in det['track_history']:
                                    px, py = point
                                    px = int(px * scale_x)
                                    py = int(py * scale_y)
                                    points.append((px, py))
                                
                                for i in range(1, len(points)):
                                    thickness = max(1, int(3 * (i / len(points))))
                                    cv2.line(web_frame, points[i-1], points[i], color, thickness)
                        except Exception as e:
                            logger.error(f"Ошибка отрисовки детекции: {e}")
                            continue
                    
                    # Добавляем информацию о парковке
                    web_frame = self._draw_parking_info(web_frame)
                    
                    # Обновление кадра
                    with self._frame_lock:
                        self._current_output_frame = web_frame.copy()
                        self._current_detections = tracked_detections.copy()
                    
                    self.processed_frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка обработки: {e}")
                time.sleep(0.1)

    def start_web_server(self):
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Parking Lot Car Counter</title>
                <meta charset="utf-8">
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
                    .controls {
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        background: rgba(0,0,0,0.7);
                        padding: 15px;
                        border-radius: 10px;
                        color: white;
                        z-index: 1000;
                        min-width: 250px;
                    }
                    .controls input, .controls button {
                        margin: 5px;
                        padding: 8px;
                        border: none;
                        border-radius: 5px;
                        width: 90%;
                    }
                    .controls button {
                        background: #4CAF50;
                        color: white;
                        cursor: pointer;
                    }
                    .controls button:hover {
                        background: #45a049;
                    }
                    .stats {
                        margin-top: 10px;
                        font-size: 14px;
                    }
                    .stats p {
                        margin: 5px 0;
                    }
                </style>
            </head>
            <body>
                <div class="controls">
                    <input type="number" id="initialCount" placeholder="Начальное количество" min="0">
                    <button onclick="setInitialCount()">Установить</button>
                    <button onclick="resetCounters()">Сбросить счетчики</button>
                    <div class="stats" id="stats">
                        <p>Въехало: <span id="inCount">0</span></p>
                        <p>Выехало: <span id="outCount">0</span></p>
                        <p>Сейчас: <span id="currentCount">0</span></p>
                        <p>Начальное: <span id="initialCountDisplay">0</span></p>
                    </div>
                </div>
                <img id="video" src="/video_feed">

                <script>
                    function refreshVideo() {
                        const video = document.getElementById('video');
                        video.src = '/video_feed?t=' + new Date().getTime();
                    }

                    function setInitialCount() {
                        const count = document.getElementById('initialCount').value;
                        if (count === '') {
                            alert('Введите количество машин');
                            return;
                        }
                        
                        fetch('/set_initial', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({count: parseInt(count)})
                        })
                        .then(response => response.json())
                        .then(data => {
                            alert(data.message);
                            updateStats();
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('Ошибка при установке начального количества');
                        });
                    }

                    function resetCounters() {
                        fetch('/reset')
                        .then(response => response.json())
                        .then(data => {
                            alert('Счетчики сброшены');
                            updateStats();
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('Ошибка при сбросе счетчиков');
                        });
                    }

                    function updateStats() {
                        fetch('/stats')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('inCount').textContent = data.vehicles_in;
                            document.getElementById('outCount').textContent = data.vehicles_out;
                            document.getElementById('currentCount').textContent = data.current_vehicles;
                            document.getElementById('initialCountDisplay').textContent = data.initial_count;
                        });
                    }

                    // Обновляем статистику каждые 3 секунды
                    setInterval(updateStats, 3000);
                    
                    // Обновляем видео каждые 5 минут для надежности
                    setInterval(refreshVideo, 300000);

                    // Автоматический рефреш при ошибках
                    document.getElementById('video').onerror = function() {
                        setTimeout(refreshVideo, 1000);
                    };

                    // Первоначальная загрузка статистики
                    updateStats();
                </script>
            </body>
            </html>
            """
        
        @app.route('/video_feed')
        def video_feed():
            def generate():
                target_fps = 10
                frame_interval = 1.0 / target_fps
                last_frame_time = 0
                
                while True:
                    try:
                        current_time = time.time()
                        if current_time - last_frame_time >= frame_interval:
                            frame, detections = self.get_latest_frame()
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.web_quality]
                            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
                            if success:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + 
                                       encoded_image.tobytes() + b'\r\n')
                                last_frame_time = current_time
                        time.sleep(0.001)
                    except Exception as e:
                        logger.error(f"Ошибка в видеопотоке: {e}")
                        time.sleep(0.1)
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/stats')
        def stats():
            elapsed = time.time() - self.start_time
            fps = self.processed_frame_count / elapsed if elapsed > 0 else 0
            
            return jsonify({
                'vehicles_in': self.parking_tracker.vehicles_in,
                'vehicles_out': self.parking_tracker.vehicles_out,
                'current_vehicles': self.parking_tracker.current_vehicles,
                'initial_count': self.parking_tracker.initial_count,
                'fps': round(fps, 1),
                'processed_frames': self.processed_frame_count,
                'uptime': round(elapsed, 1)
            })
        
        @app.route('/reset')
        def reset_counters():
            self.parking_tracker.vehicles_in = 0
            self.parking_tracker.vehicles_out = 0
            # Не сбрасываем initial_count и current_vehicles
            self.parking_tracker.current_vehicles = self.parking_tracker.initial_count
            return jsonify({"status": "counters reset"})
        
        @app.route('/set_initial', methods=['POST'])
        def set_initial_count():
            try:
                data = request.get_json()
                if not data or 'count' not in data:
                    return jsonify({"error": "No count provided"}), 400
                
                count = int(data['count'])
                if count < 0:
                    return jsonify({"error": "Count must be positive"}), 400
                
                self.parking_tracker.set_initial_count(count)
                return jsonify({"message": f"Initial count set to {count}"})
            except Exception as e:
                logger.error(f"Error setting initial count: {e}")
                return jsonify({"error": str(e)}), 500
        
        logger.info(f"🌐 Запуск веб-сервера парковки на http://{self.config.web_host}:{self.config.web_port}")
        app.run(host=self.config.web_host, port=self.config.web_port, threaded=True, debug=False)

    def start_processing(self):
        if not self.start_ffmpeg():
            return False
        
        if not self.load_yolo_model():
            return False
        
        self.running = True
        
        with self._frame_lock:
            self._current_output_frame = self._create_info_frame("Initializing Parking Lot Monitor...")
        
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        
        capture_thread.start()
        time.sleep(3)
        process_thread.start()
        
        logger.info("✅ Система подсчета парковки запущена")
        return True

    def start(self):
        if not self.start_processing():
            return False
        
        self.start_web_server()
        return True

    def stop(self):
        self.running = False
        if hasattr(self, 'pipe'):
            self.pipe.terminate()

def main():
    config = ParkingConfig()
    processor = ParkingLotProcessor(config)
    
    try:
        if processor.start():
            logger.info("✅ Система подсчета автомобилей на парковке запущена")
            logger.info("🚗 Настройте counting_line в конфиге под вашу камеру")
            logger.info("💡 Используйте веб-интерфейс для установки начального количества машин")
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