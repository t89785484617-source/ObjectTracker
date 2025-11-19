#!/usr/bin/env python3
"""
Production RTSP to YOLO Processor - ADVANCED OBJECT TRACKING with ANALYTICS
CAR-ONLY DETECTION VERSION WITH CROSSING LINE AND REAL-TIME STATISTICS
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

# Настройка основного логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rtsp_yolo_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# НАСТРОЙКА ЛОГГЕРА ДЛЯ АНАЛИТИКИ ТРЕКИНГА
analytics_logger = logging.getLogger('tracking_analytics')
analytics_logger.setLevel(logging.INFO)
analytics_handler = logging.FileHandler('tracking_analytics.log')
analytics_handler.setFormatter(logging.Formatter('%(message)s'))
analytics_logger.addHandler(analytics_handler)
analytics_logger.propagate = False

class Config:
    def __init__(self):
        self.rtsp_url = "rtsp://admin:Jaquio@172.30.0.68:554/live/main"
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
        self.tracker_max_age = 30
        self.tracker_min_hits = 3
        self.tracker_iou_threshold = 0.4
        self.tracker_appearance_weight = 0.7
        self.tracker_velocity_weight = 0.3
        
        # НАСТРОЙКИ ЛОГИРОВАНИЯ АНАЛИТИКИ
        self.analytics_log_interval = 5  # секунды между логами аналитики
        self.detailed_log_interval = 30  # секунды для детального лога
        
        # КЛАССЫ ДЛЯ ДЕТЕКЦИИ (ТОЛЬКО АВТОМОБИЛИ)
        self.target_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck в COCO dataset
        
        # НАСТРОЙКИ ЛИНИИ ПЕРЕСЕЧЕНИЯ
        self.crossing_line_y_ratio = 0.5  # Позиция линии (0.5 = центр)

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
    """Трекаемый объект с улучшенной стабильностью ID и отслеживанием пересечения линии"""
    
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
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
        
        # Сразу увеличиваем при создании
        self.age += 1
        self.hit_streak += 1
        
        # Статус пересечения линии
        self.has_crossed_line = False
        self.crossing_direction = None  # 'entering' или 'exiting'
        self.last_position_y = self._get_center_y()
        
        # Визуальные особенности (упрощенные)
        self.appearance_features = self._extract_appearance(detection['bbox'])
        
        self.config = config
    
    def _get_center_y(self):
        """Получение Y-координаты центра объекта"""
        bbox = self.kalman.get_bbox()
        return (bbox[1] + bbox[3]) / 2
    
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
        self.age += 1
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
        self.update_track_history()
        
        # Обновление визуальных особенностей
        self.appearance_features = self._extract_appearance(detection['bbox'])
    
    def check_line_crossing(self, line_y):
        """Проверка пересечения линии и обновление статистики"""
        current_y = self._get_center_y()
        
        # Если объект уже пересек линию, не считаем повторно
        if self.has_crossed_line:
            self.last_position_y = current_y
            return False
        
        # Проверяем пересечение линии
        if (self.last_position_y <= line_y and current_y > line_y) or \
           (self.last_position_y >= line_y and current_y < line_y):
            
            # Определяем направление
            if current_y > line_y:
                self.crossing_direction = 'exiting'  # сверху вниз - выезжает
            else:
                self.crossing_direction = 'entering'  # снизу вверх - заезжает
            
            self.has_crossed_line = True
            self.last_position_y = current_y
            return True
        
        self.last_position_y = current_y
        return False
    
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
    """Продвинутый трекер объектов с стабильными ID и отслеживанием пересечений"""
    
    def __init__(self, config):
        self.config = config
        self.next_object_id = 1
        self.tracked_objects = OrderedDict()  # object_id -> TrackedObject
        self.frames_since_update = 0
        
        # Статистика пересечений
        self.entering_count = 0
        self.exiting_count = 0
    
    def update(self, detections, line_y):
        """Обновление трекера с новыми детекциями и проверка пересечений"""
        self.frames_since_update += 1
        
        # Предсказание позиций для всех существующих объектов
        for obj in self.tracked_objects.values():
            obj.predict()
            
            # Проверка пересечения линии для каждого объекта
            if obj.check_line_crossing(line_y):
                if obj.crossing_direction == 'entering':
                    self.entering_count += 1
                    logger.info(f"🚗 ВЪЕЗД: Автомобиль ID:{obj.object_id} заезжает (всего: {self.entering_count})")
                else:
                    self.exiting_count += 1
                    logger.info(f"🚗 ВЫЕЗД: Автомобиль ID:{obj.object_id} выезжает (всего: {self.exiting_count})")
        
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
                detection['has_crossed_line'] = obj.has_crossed_line
                detection['crossing_direction'] = obj.crossing_direction
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
    
    def get_crossing_stats(self):
        """Получение статистики пересечений"""
        return {
            'entering': self.entering_count,
            'exiting': self.exiting_count,
            'total': self.entering_count + self.exiting_count
        }

class RTSPYOLOProcessor:
    def __init__(self, config):
        self.config = config
        self.frame_size = config.capture_width * config.capture_height * 3
        
        # Инициализация улучшенного трекера
        self.object_tracker = AdvancedObjectTracker(config)
        
        # Позиция линии пересечения (в координатах processing frame)
        self.crossing_line_y = int(self.config.processing_height * self.config.crossing_line_y_ratio)
        
        # ЕДИНСТВЕННЫЙ буфер для веб-вывода
        self.output_buffer = queue.Queue(maxsize=1)
        
        # Отдельный буфер для обработки
        self.processing_buffer = queue.Queue(maxsize=5)
        
        self.running = False
        self.capture_frame_count = 0
        self.processed_frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # Для аналитики
        self.last_analytics_log_time = 0
        self.last_detailed_log_time = 0
        self.tracking_stats = {
            'total_tracks_created': 0,
            'total_tracks_lost': 0,
            'max_track_age': 0,
            'max_track_hits': 0,
            'class_distribution': {},
            'track_quality_history': []
        }
        
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
            logger.info(f"🎯 Режим детекции: ТОЛЬКО АВТОМОБИЛИ (классы {self.config.target_classes})")
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

    def _log_tracking_analytics(self):
        """Логирование аналитики трекинга"""
        current_time = time.time()
        
        # Логируем базовую аналитику каждые N секунд
        if current_time - self.last_analytics_log_time >= self.config.analytics_log_interval:
            active_tracks = len(self.object_tracker.tracked_objects)
            active_detections = len(self._current_detections)
            crossing_stats = self.object_tracker.get_crossing_stats()
            
            # Собираем статистику по активным трекам
            track_qualities = []
            class_distribution = {}
            
            for obj_id, obj in self.object_tracker.tracked_objects.items():
                quality = obj.hit_streak / obj.age if obj.age > 0 else 1.0
                track_qualities.append(quality)
                
                # Распределение по классам
                class_name = obj.class_name
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                
                # Обновляем максимальные значения
                self.tracking_stats['max_track_age'] = max(self.tracking_stats['max_track_age'], obj.age)
                self.tracking_stats['max_track_hits'] = max(self.tracking_stats['max_track_hits'], obj.hit_streak)
            
            avg_quality = np.mean(track_qualities) if track_qualities else 0
            
            # Логируем базовую аналитику с учетом статистики пересечений
            analytics_data = {
                'timestamp': datetime.now().isoformat(),
                'active_tracks': active_tracks,
                'active_detections': active_detections,
                'avg_track_quality': round(avg_quality, 3),
                'min_track_quality': round(min(track_qualities), 3) if track_qualities else 0,
                'max_track_quality': round(max(track_qualities), 3) if track_qualities else 0,
                'class_distribution': class_distribution,
                'total_processed_frames': self.processed_frame_count,
                'total_detections': self.detection_count,
                'crossing_stats': crossing_stats
            }
            
            analytics_logger.info(json.dumps(analytics_data))
            self.last_analytics_log_time = current_time
            
            # Сохраняем историю качества
            self.tracking_stats['track_quality_history'].append({
                'time': current_time,
                'avg_quality': avg_quality,
                'active_tracks': active_tracks,
                'crossing_stats': crossing_stats
            })
            
            # Ограничиваем размер истории
            if len(self.tracking_stats['track_quality_history']) > 1000:
                self.tracking_stats['track_quality_history'] = self.tracking_stats['track_quality_history'][-1000:]
        
        # Детальное логирование каждые 30 секунд
        if current_time - self.last_detailed_log_time >= self.config.detailed_log_interval:
            self._log_detailed_tracking_info()
            self.last_detailed_log_time = current_time

    def _log_detailed_tracking_info(self):
        """Детальное логирование информации о треках"""
        crossing_stats = self.object_tracker.get_crossing_stats()
        
        detailed_info = {
            'timestamp': datetime.now().isoformat(),
            'total_tracks_created': self.tracking_stats['total_tracks_created'],
            'total_tracks_lost': self.tracking_stats['total_tracks_lost'],
            'max_track_age': self.tracking_stats['max_track_age'],
            'max_track_hits': self.tracking_stats['max_track_hits'],
            'crossing_stats': crossing_stats,
            'current_tracks': []
        }
        
        for obj_id, obj in self.object_tracker.tracked_objects.items():
            track_info = {
                'id': obj_id,
                'class': obj.class_name,
                'age': obj.age,
                'hits': obj.hit_streak,
                'quality': round(obj.hit_streak / obj.age, 3) if obj.age > 0 else 1.0,
                'time_since_update': obj.time_since_update,
                'current_confidence': obj.confidence,
                'has_crossed_line': obj.has_crossed_line,
                'crossing_direction': obj.crossing_direction
            }
            detailed_info['current_tracks'].append(track_info)
        
        # Логируем в отдельный файл для детального анализа
        with open('detailed_tracking_analysis.log', 'a') as f:
            f.write(json.dumps(detailed_info) + '\n')
        
        logger.info(f"📊 Детальная аналитика: {len(detailed_info['current_tracks'])} активных треков, "
                   f"макс. возраст: {self.tracking_stats['max_track_age']}, "
                   f"макс. hits: {self.tracking_stats['max_track_hits']}, "
                   f"въездов: {crossing_stats['entering']}, выездов: {crossing_stats['exiting']}")

    def _update_tracking_stats(self, detections_before, detections_after):
        """Обновление статистики трекинга после обработки кадра"""
        # Обновляем счетчики созданных/потерянных треков
        current_track_ids = set(obj.object_id for obj in self.object_tracker.tracked_objects.values())
        previous_track_ids = set(det['object_id'] for det in detections_before) if detections_before else set()
        
        new_tracks = current_track_ids - previous_track_ids
        lost_tracks = previous_track_ids - current_track_ids
        
        self.tracking_stats['total_tracks_created'] += len(new_tracks)
        self.tracking_stats['total_tracks_lost'] += len(lost_tracks)
        
        # Логируем создание новых треков
        for track_id in new_tracks:
            obj = self.object_tracker.tracked_objects[track_id]
            logger.info(f"🆕 Новый трек: ID:{track_id} {obj.class_name} (confidence: {obj.confidence:.2f})")
        
        # Логируем потерю треков
        for track_id in lost_tracks:
            logger.info(f"❌ Потерян трек: ID:{track_id}")

    def process_frames(self):
        """Обработка кадров с YOLO - с улучшенным трекингом, аналитикой и отслеживанием пересечений"""
        logger.info("🔍 Запуск обработки YOLO с улучшенным трекингом (ТОЛЬКО АВТОМОБИЛИ)")
        logger.info(f"📏 Линия пересечения установлена на Y={self.crossing_line_y} (координаты processing frame)")
        
        frame_counter = 0
        
        while self.running:
            try:
                # Берем кадр из буфера обработки
                frame = self.processing_buffer.get(timeout=1.0)
                frame_counter += 1
                
                # Сохраняем предыдущие детекции для анализа изменений
                previous_detections = self._current_detections.copy()
                
                # Обрабатываем каждый N-й кадр
                if frame_counter % self.config.process_every_n == 0:
                    # Подготовка кадра для YOLO
                    processing_frame = self.resize_frame_proportional(
                        frame, 
                        self.config.processing_width, 
                        self.config.processing_height
                    )
                    
                    # YOLO обработка ТОЛЬКО ДЛЯ АВТОМОБИЛЕЙ
                    results = self.model(processing_frame, 
                                       conf=self.config.confidence_threshold,
                                       classes=self.config.target_classes,  # ФИЛЬТРАЦИЯ ПО КЛАССАМ
                                       verbose=False)
                    
                    # Извлечение детекций (только автомобили)
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls[0])
                                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА (на всякий случай)
                                if cls not in self.config.target_classes:
                                    continue
                                    
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
                    
                    # ОБНОВЛЕНИЕ УЛУЧШЕННОГО ТРЕКЕРА С ПЕРЕДАЧЕЙ ПОЗИЦИИ ЛИНИИ
                    tracked_detections = self.object_tracker.update(detections, self.crossing_line_y)
                    
                    # ОБНОВЛЯЕМ СТАТИСТИКУ ТРЕКИНГА
                    self._update_tracking_stats(previous_detections, tracked_detections)
                    
                    # ЛОГИРУЕМ АНАЛИТИКУ
                    self._log_tracking_analytics()
                    
                    # Создание кадра для веб-вывода
                    web_frame = self.resize_frame_proportional(
                        frame,
                        self.config.web_width,
                        self.config.web_height
                    )
                    
                    # Масштабирование bounding boxes и линии
                    scale_x = self.config.web_width / self.config.processing_width
                    scale_y = self.config.web_height / self.config.processing_height
                    web_line_y = int(self.crossing_line_y * scale_y)
                    
                    # Отрисовка ЛИНИИ ПЕРЕСЕЧЕНИЯ
                    cv2.line(web_frame, (0, web_line_y), (self.config.web_width, web_line_y), 
                            (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Подпись для линии
                    cv2.putText(web_frame, "CROSSING LINE", (10, web_line_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Статистика пересечений на кадре
                    crossing_stats = self.object_tracker.get_crossing_stats()
                    stats_text = f"ENTERING: {crossing_stats['entering']} | EXITING: {crossing_stats['exiting']} | TOTAL: {crossing_stats['total']}"
                    cv2.putText(web_frame, stats_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
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
                        quality = hit_streak / age if age > 0 else 1.0
                        label = f"ID:{object_id} {det['class_name']} {det['confidence']:.2f}"
                        sub_label = f"Age:{age} Hits:{hit_streak} Q:{quality:.2f}"
                        
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

    def start_web_server(self):
        """Запуск веб-сервера с фиксированным FPS и аналитикой"""
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>VisionGuard RTSP - Advanced CAR Tracking</title>
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
                </style>
            </head>
            <body>
                <img id="video" src="/video_feed">

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
            
            # Рассчитываем качество трекинга
            track_qualities = []
            for obj in self.object_tracker.tracked_objects.values():
                if obj.age > 0:
                    track_qualities.append(obj.hit_streak / obj.age)
            
            avg_quality = np.mean(track_qualities) if track_qualities else 0
            
            crossing_stats = self.object_tracker.get_crossing_stats()
            
            return {
                'objects_count': len(self._current_detections),
                'fps': round(fps, 1),
                'total_tracks': len(self.object_tracker.tracked_objects),
                'processed_frames': self.processed_frame_count,
                'total_detections': self.detection_count,
                'avg_track_quality': round(avg_quality, 3),
                'tracks_created': self.tracking_stats['total_tracks_created'],
                'tracks_lost': self.tracking_stats['total_tracks_lost'],
                'max_track_age': self.tracking_stats['max_track_age'],
                'max_track_hits': self.tracking_stats['max_track_hits'],
                'crossing_stats': crossing_stats
            }
        
        @app.route('/analytics')
        def analytics():
            """Расширенная аналитика трекинга"""
            current_tracks = []
            for obj_id, obj in self.object_tracker.tracked_objects.items():
                quality = obj.hit_streak / obj.age if obj.age > 0 else 1.0
                current_tracks.append({
                    'id': obj_id,
                    'class': obj.class_name,
                    'age': obj.age,
                    'hits': obj.hit_streak,
                    'quality': round(quality, 3),
                    'time_since_update': obj.time_since_update,
                    'confidence': round(obj.confidence, 3),
                    'has_crossed_line': obj.has_crossed_line,
                    'crossing_direction': obj.crossing_direction
                })
            
            # Сортируем по качеству
            current_tracks.sort(key=lambda x: x['quality'], reverse=True)
            
            crossing_stats = self.object_tracker.get_crossing_stats()
            
            return {
                'current_tracks': current_tracks,
                'tracking_stats': self.tracking_stats,
                'crossing_stats': crossing_stats,
                'system_uptime': round(time.time() - self.start_time, 1)
            }
        
        logger.info(f"🌐 Запуск веб-сервера на http://{self.config.web_host}:{self.config.web_port}")
        logger.info("📊 Доступна аналитика по адресу: /stats и /analytics")
        app.run(host=self.config.web_host, port=self.config.web_port, threaded=True, debug=False)

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
            logger.info("✅ Система запущена с улучшенным трекингом АВТОМОБИЛЕЙ")
            logger.info("🎯 Режим: ТОЛЬКО автомобили (car, motorcycle, bus, truck)")
            logger.info("📏 Линия пересечения: центр кадра (сверху вниз = выезд, снизу вверх = въезд)")
            logger.info("📊 Логи аналитики сохраняются в tracking_analytics.log")
            logger.info("📈 Детальная аналитика в detailed_tracking_analysis.log")
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