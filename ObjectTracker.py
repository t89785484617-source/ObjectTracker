#!/usr/bin/env python3
"""
Production RTSP to YOLO Processor - ADVANCED OBJECT TRACKING with ANALYTICS
CAR-ONLY DETECTION VERSION WITH IMPROVED TRACKING STABILITY
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
        self.web_port = 8002
        self.web_quality = 60
        
        # УЛУЧШЕННЫЕ НАСТРОЙКИ ТРЕКЕРА ДЛЯ СТАБИЛЬНОСТИ
        self.tracker_max_age = 45
        self.tracker_min_hits = 4
        self.tracker_iou_threshold = 0.3
        self.tracker_appearance_weight = 0.6
        self.tracker_velocity_weight = 0.4
        
        # НАСТРОЙКИ ЛОГИРОВАНИЯ АНАЛИТИКИ
        self.analytics_log_interval = 5
        self.detailed_log_interval = 30
        
        # КЛАССЫ ДЛЯ ДЕТЕКЦИИ (ТОЛЬКО АВТОМОБИЛИ)
        self.target_classes = [2, 3, 5, 7]
        
        # НАСТРОЙКИ ВЕРТИКАЛЬНОЙ ЛИНИИ ПЕРЕСЕЧЕНИЯ
        self.crossing_line_x_ratio = 0.5
        self.crossing_tolerance = 15
        
        # ЗАЩИТА ОТ ДУБЛИРОВАНИЯ ПЕРЕСЕЧЕНИЙ
        self.crossing_cooldown = 300
        self.min_crossing_confidence = 0.6

class KalmanFilter:
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

class TrackedObject:
    def __init__(self, object_id, detection, config):
        self.object_id = object_id
        self.detection = detection
        self.class_name = detection['class_name']
        self.confidence = detection['confidence']
        
        # ИСПРАВЛЕНИЕ: Сохраняем config как атрибут объекта
        self.config = config
        
        self.kalman = KalmanFilter()
        self.kalman.init(detection['bbox'])
        
        self.track_history = deque(maxlen=50)
        self.update_track_history()
        
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
        
        self.age += 1
        self.hit_streak += 1
        
        self.has_crossed_line = False
        self.crossing_direction = None
        self.last_position_x = self._get_center_x()
        self.crossing_verified = False
        self.crossing_frame = 0
        self.crossing_cooldown_counter = 0
        
        self.position_history = deque(maxlen=8)
        self.position_history.append(self.last_position_x)
        
        self.appearance_features = self._extract_appearance(detection['bbox'])
        self.stable_features = self._extract_stable_features(detection['bbox'])
    
    def _get_center_x(self):
        bbox = self.kalman.get_bbox()
        return (bbox[0] + bbox[2]) / 2
    
    def _extract_appearance(self, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        return np.array([w, h, aspect_ratio, area])
    
    def _extract_stable_features(self, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        # ИСПРАВЛЕНИЕ: Используем self.config вместо прямого обращения
        normalized_w = w / self.config.processing_width
        normalized_h = h / self.config.processing_height
        return np.array([aspect_ratio, normalized_w, normalized_h, area])
    
    def update_track_history(self):
        bbox = self.kalman.get_bbox()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.track_history.append((cx, cy))
    
    def predict(self):
        predicted_bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        self.update_track_history()
        
        if self.has_crossed_line:
            self.crossing_cooldown_counter += 1
            
        return predicted_bbox
    
    def update(self, detection):
        self.detection = detection
        self.confidence = detection['confidence']
        self.kalman.update(detection['bbox'])
        self.hit_streak += 1
        self.time_since_update = 0
        self.update_track_history()
        
        self.appearance_features = self._extract_appearance(detection['bbox'])
        self.stable_features = self._extract_stable_features(detection['bbox'])
    
    def can_cross_again(self):
        if not self.has_crossed_line:
            return True
        return self.crossing_cooldown_counter > self.config.crossing_cooldown
    
    def check_line_crossing(self, line_x):
        if self.has_crossed_line and not self.can_cross_again():
            return False
            
        current_x = self._get_center_x()
        self.position_history.append(current_x)
        
        if len(self.position_history) < 3:
            self.last_position_x = current_x
            return False
        
        oldest_x = self.position_history[0]
        direction = "right" if current_x > oldest_x else "left"
        
        line_crossed = False
        crossing_detected = None
        
        for i in range(1, len(self.position_history)):
            prev_x = self.position_history[i-1]
            curr_x = self.position_history[i]
            
            if (prev_x <= line_x + self.config.crossing_tolerance and 
                curr_x >= line_x - self.config.crossing_tolerance) or \
               (prev_x >= line_x - self.config.crossing_tolerance and 
                curr_x <= line_x + self.config.crossing_tolerance):
                
                if curr_x > prev_x:
                    crossing_detected = 'exiting'
                else:
                    crossing_detected = 'entering'
                
                line_crossed = True
                break
        
        if line_crossed and crossing_detected:
            if (crossing_detected == 'exiting' and direction == "right") or \
               (crossing_detected == 'entering' and direction == "left"):
                
                if self.confidence >= self.config.min_crossing_confidence:
                    self.has_crossed_line = True
                    self.crossing_direction = crossing_detected
                    self.crossing_verified = True
                    self.crossing_frame = self.age
                    self.crossing_cooldown_counter = 0
                    
                    logger.info(f"🚗 ПЕРЕСЕЧЕНИЕ: ID:{self.object_id} направление: {crossing_detected} "
                              f"(confidence: {self.confidence:.2f}, возраст: {self.age})")
                    return True
        
        self.last_position_x = current_x
        return False
    
    def similarity_score(self, detection):
        bbox1 = self.kalman.get_bbox()
        bbox2 = detection['bbox']
        
        iou = self._calculate_iou(bbox1, bbox2)
        
        class_similarity = 1.0 if self.class_name == detection['class_name'] else 0.0
        
        features1 = self.stable_features
        features2 = self._extract_stable_features(bbox2)
        size_similarity = 1.0 - min(1.0, np.linalg.norm(features1 - features2))
        
        pred_bbox = self.kalman.get_bbox()
        center1 = np.array([(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        distance = np.linalg.norm(center1 - center2)
        position_similarity = max(0, 1.0 - distance / 100.0)
        
        motion_similarity = (iou * 0.6 + position_similarity * 0.4) * self.config.tracker_velocity_weight
        appearance_similarity = (class_similarity * 0.4 + size_similarity * 0.6) * self.config.tracker_appearance_weight
        
        return motion_similarity + appearance_similarity
    
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

class AdvancedObjectTracker:
    def __init__(self, config):
        self.config = config
        self.next_object_id = 1
        self.tracked_objects = OrderedDict()
        self.frames_since_update = 0
        self.last_crossing_time = 0
        
        self.entering_count = 0
        self.exiting_count = 0
        
        self.recent_crossings = {}
        self.crossing_suppression_time = 10.0
        
        self.tracking_history = deque(maxlen=100)
    
    def update(self, detections, line_x):
        self.frames_since_update += 1
        
        for obj in self.tracked_objects.values():
            obj.predict()
            
            current_time = time.time()
            if (obj.object_id not in self.recent_crossings or 
                current_time - self.recent_crossings[obj.object_id] > self.crossing_suppression_time):
                
                if obj.check_line_crossing(line_x):
                    self._register_crossing(obj)
        
        active_detections = self._match_detections(detections)
        
        self._update_unmatched_tracks()
        
        self._create_new_tracks(detections, active_detections)
        
        return self._get_active_detections()
    
    def _match_detections(self, detections):
        if not detections or not self.tracked_objects:
            return set()
        
        filtered_detections = [d for d in detections if d['confidence'] > 0.4]
        
        if not filtered_detections:
            return set()
        
        similarity_matrix = self._create_similarity_matrix(filtered_detections)
        matched_pairs = self._hungarian_matching(similarity_matrix)
        
        matched_detections = set()
        matched_tracks = set()
        
        for det_idx, track_idx in matched_pairs:
            if similarity_matrix[det_idx][track_idx] > self.config.tracker_iou_threshold:
                track_id = list(self.tracked_objects.keys())[track_idx]
                detection = filtered_detections[det_idx]
                
                self.tracked_objects[track_id].update(detection)
                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)
                
                logger.debug(f"✅ Сопоставление: ID:{track_id} с детекцией {det_idx} "
                           f"(score: {similarity_matrix[det_idx][track_idx]:.3f})")
        
        return matched_detections
    
    def _create_similarity_matrix(self, detections):
        track_ids = list(self.tracked_objects.keys())
        similarity_matrix = np.zeros((len(detections), len(track_ids)))
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                obj = self.tracked_objects[track_id]
                similarity_matrix[det_idx][track_idx] = obj.similarity_score(detection)
        
        return similarity_matrix
    
    def _hungarian_matching(self, cost_matrix):
        cost_matrix = 1 - cost_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))
    
    def _update_unmatched_tracks(self):
        tracks_to_remove = []
        
        for track_id, obj in self.tracked_objects.items():
            if obj.time_since_update > 0:
                if obj.time_since_update > self.config.tracker_max_age:
                    tracks_to_remove.append(track_id)
                elif obj.time_since_update > self.config.tracker_max_age // 2:
                    obj.confidence *= 0.95
        
        for track_id in tracks_to_remove:
            if track_id in self.tracked_objects:
                logger.debug(f"🗑️ Удален трек ID:{track_id} (age: {self.tracked_objects[track_id].age})")
                del self.tracked_objects[track_id]
    
    def _create_new_tracks(self, detections, matched_detections):
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                if detection['confidence'] > 0.5:
                    if not self._is_duplicate_detection(detection):
                        self._create_new_track(detection)
    
    def _is_duplicate_detection(self, detection):
        for obj in self.tracked_objects.values():
            similarity = obj.similarity_score(detection)
            if similarity > self.config.tracker_iou_threshold * 0.8:
                logger.debug(f"🚫 Подавлен дубликат детекции для ID:{obj.object_id} (score: {similarity:.3f})")
                return True
        return False
    
    def _create_new_track(self, detection):
        object_id = self.next_object_id
        self.tracked_objects[object_id] = TrackedObject(object_id, detection, self.config)
        logger.info(f"🆕 Новый трек: ID:{object_id} {detection['class_name']} "
                   f"(confidence: {detection['confidence']:.2f})")
        self.next_object_id += 1
    
    def _register_crossing(self, obj):
        current_time = time.time()
        
        if (obj.object_id in self.recent_crossings and 
            current_time - self.recent_crossings[obj.object_id] < self.crossing_suppression_time):
            logger.debug(f"🚫 Подавлено повторное пересечение ID:{obj.object_id}")
            return
        
        self.recent_crossings[obj.object_id] = current_time
        
        if obj.crossing_direction == 'entering':
            self.entering_count += 1
            logger.info(f"🚗 ВЪЕЗД: Автомобиль ID:{obj.object_id} заезжает СПРАВА НАЛЕВО "
                       f"(всего: {self.entering_count}, confidence: {obj.confidence:.2f})")
        else:
            self.exiting_count += 1
            logger.info(f"🚗 ВЫЕЗД: Автомобиль ID:{obj.object_id} выезжает СЛЕВА НАПРАВО "
                       f"(всего: {self.exiting_count}, confidence: {obj.confidence:.2f})")
    
    def _get_active_detections(self):
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
    
    def get_crossing_stats(self):
        return {
            'entering': self.entering_count,
            'exiting': self.exiting_count,
            'total': self.entering_count + self.exiting_count
        }

class RTSPYOLOProcessor:
    def __init__(self, config):
        self.config = config
        self.frame_size = config.capture_width * config.capture_height * 3
        
        self.object_tracker = AdvancedObjectTracker(config)
        
        self.crossing_line_x = int(self.config.processing_width * self.config.crossing_line_x_ratio)
        
        self.output_buffer = queue.Queue(maxsize=1)
        self.processing_buffer = queue.Queue(maxsize=5)
        
        self.running = False
        self.capture_frame_count = 0
        self.processed_frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
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
        
        self._current_output_frame = self._create_info_frame("Starting...")
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
                        
                        if not self.processing_buffer.full():
                            self.processing_buffer.put(frame)
                        
                        self.capture_frame_count += 1
                        consecutive_errors = 0
                        
                    else:
                        logger.warning(f"Неполный кадр: {len(raw_frame)}/{self.frame_size}")
                        consecutive_errors += 1
                else:
                    logger.warning("Таймаут чтения кадра")
                    consecutive_errors += 1
                
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

    def _log_tracking_analytics(self):
        current_time = time.time()
        
        if current_time - self.last_analytics_log_time >= self.config.analytics_log_interval:
            active_tracks = len(self.object_tracker.tracked_objects)
            active_detections = len(self._current_detections)
            crossing_stats = self.object_tracker.get_crossing_stats()
            
            track_qualities = []
            class_distribution = {}
            
            for obj_id, obj in self.object_tracker.tracked_objects.items():
                quality = obj.hit_streak / obj.age if obj.age > 0 else 1.0
                track_qualities.append(quality)
                
                class_name = obj.class_name
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                
                self.tracking_stats['max_track_age'] = max(self.tracking_stats['max_track_age'], obj.age)
                self.tracking_stats['max_track_hits'] = max(self.tracking_stats['max_track_hits'], obj.hit_streak)
            
            avg_quality = np.mean(track_qualities) if track_qualities else 0
            
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
            
            self.tracking_stats['track_quality_history'].append({
                'time': current_time,
                'avg_quality': avg_quality,
                'active_tracks': active_tracks,
                'crossing_stats': crossing_stats
            })
            
            if len(self.tracking_stats['track_quality_history']) > 1000:
                self.tracking_stats['track_quality_history'] = self.tracking_stats['track_quality_history'][-1000:]
        
        if current_time - self.last_detailed_log_time >= self.config.detailed_log_interval:
            self._log_detailed_tracking_info()
            self.last_detailed_log_time = current_time

    def _log_detailed_tracking_info(self):
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
        
        with open('detailed_tracking_analysis.log', 'a') as f:
            f.write(json.dumps(detailed_info) + '\n')
        
        logger.info(f"📊 Детальная аналитика: {len(detailed_info['current_tracks'])} активных треков, "
                   f"макс. возраст: {self.tracking_stats['max_track_age']}, "
                   f"макс. hits: {self.tracking_stats['max_track_hits']}, "
                   f"въездов: {crossing_stats['entering']}, выездов: {crossing_stats['exiting']}")

    def _update_tracking_stats(self, detections_before, detections_after):
        current_track_ids = set(obj.object_id for obj in self.object_tracker.tracked_objects.values())
        previous_track_ids = set(det['object_id'] for det in detections_before) if detections_before else set()
        
        new_tracks = current_track_ids - previous_track_ids
        lost_tracks = previous_track_ids - current_track_ids
        
        self.tracking_stats['total_tracks_created'] += len(new_tracks)
        self.tracking_stats['total_tracks_lost'] += len(lost_tracks)
        
        for track_id in new_tracks:
            obj = self.object_tracker.tracked_objects[track_id]
            logger.info(f"🆕 Новый трек: ID:{track_id} {obj.class_name} (confidence: {obj.confidence:.2f})")
        
        for track_id in lost_tracks:
            logger.info(f"❌ Потерян трек: ID:{track_id}")

    def process_frames(self):
        logger.info("🔍 Запуск обработки YOLO с УЛУЧШЕННЫМ трекингом")
        logger.info(f"🎯 Защита от дублирования: {self.config.crossing_cooldown} кадров")
        logger.info(f"📏 Линия пересечения: X={self.crossing_line_x}")
        
        frame_counter = 0
        
        while self.running:
            try:
                frame = self.processing_buffer.get(timeout=1.0)
                frame_counter += 1
                
                previous_detections = self._current_detections.copy()
                
                if frame_counter % self.config.process_every_n == 0:
                    processing_frame = self.resize_frame_proportional(
                        frame, 
                        self.config.processing_width, 
                        self.config.processing_height
                    )
                    
                    results = self.model(processing_frame, 
                                       conf=self.config.confidence_threshold,
                                       classes=self.config.target_classes,
                                       verbose=False)
                    
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls[0])
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
                    
                    tracked_detections = self.object_tracker.update(detections, self.crossing_line_x)
                    
                    self._update_tracking_stats(previous_detections, tracked_detections)
                    self._log_tracking_analytics()
                    
                    web_frame = self.resize_frame_proportional(
                        frame,
                        self.config.web_width,
                        self.config.web_height
                    )
                    
                    scale_x = self.config.web_width / self.config.processing_width
                    scale_y = self.config.web_height / self.config.processing_height
                    web_line_x = int(self.crossing_line_x * scale_x)
                    
                    self._draw_crossing_line(web_frame, web_line_x)
                    self._draw_stats(web_frame, scale_x)
                    self._draw_detections(web_frame, tracked_detections, scale_x, scale_y)
                    
                    with self._frame_lock:
                        self._current_output_frame = web_frame.copy()
                        self._current_detections = tracked_detections.copy()
                    
                    self.processed_frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка обработки: {e}")
                time.sleep(0.1)

    def _draw_crossing_line(self, frame, line_x):
        cv2.line(frame, (line_x, 0), (line_x, self.config.web_height), 
                (0, 255, 255), 3, cv2.LINE_AA)
        
        cv2.putText(frame, "CROSSING LINE", (line_x + 10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.arrowedLine(frame, (line_x + 50, 60), (line_x - 50, 60), 
                       (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(frame, "ENTERING", (line_x + 60, 65),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.arrowedLine(frame, (line_x - 50, 90), (line_x + 50, 90), 
                       (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(frame, "EXITING", (line_x - 80, 95),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def _draw_stats(self, frame, scale_x):
        crossing_stats = self.object_tracker.get_crossing_stats()
        stats_text = f"ENTERING: {crossing_stats['entering']} | EXITING: {crossing_stats['exiting']} | TOTAL: {crossing_stats['total']}"
        cv2.putText(frame, stats_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        line_info = f"Line X: {int(self.crossing_line_x * scale_x)} (tolerance: {int(self.config.crossing_tolerance * scale_x)})"
        cv2.putText(frame, line_info, (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        protection_info = f"Anti-duplicate: {self.config.crossing_cooldown}frames"
        cv2.putText(frame, protection_info, (10, 80),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_detections(self, frame, detections, scale_x, scale_y):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y) 
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            object_id = det.get('object_id', 0)
            color = self._get_color_by_id(object_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            
            age = det.get('age', 1)
            hit_streak = det.get('hit_streak', 1)
            quality = hit_streak / age if age > 0 else 1.0
            
            crossing_status = ""
            if det.get('has_crossed_line', False):
                direction = det.get('crossing_direction', 'unknown')
                crossing_status = f" | CROSSED: {direction.upper()}"
            
            label = f"ID:{object_id} {det['class_name']} {det['confidence']:.2f}"
            sub_label = f"Age:{age} Hits:{hit_streak} Q:{quality:.2f}{crossing_status}"
            
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(frame, (x1, y1-text_height-25), 
                        (x1+text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, sub_label, (x1, y1-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if 'track_history' in det and len(det['track_history']) > 1:
                points = []
                for point in det['track_history']:
                    px, py = point
                    px = int(px * scale_x)
                    py = int(py * scale_y)
                    points.append((px, py))
                
                for i in range(1, len(points)):
                    thickness = max(1, int(3 * (i / len(points))))
                    cv2.line(frame, points[i-1], points[i], color, thickness)

    def start_web_server(self):
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

                    setInterval(refreshVideo, 300000);

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
        if not self.start_ffmpeg():
            return False
        
        if not self.load_yolo_model():
            return False
        
        self.running = True
        
        with self._frame_lock:
            self._current_output_frame = self._create_info_frame("Initializing...")
        
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        
        capture_thread.start()
        time.sleep(3)
        process_thread.start()
        
        logger.info("✅ Все потоки запущены")
        return True

    def start(self):
        if not self.start_processing():
            return False
        
        web_thread = threading.Thread(target=self.start_web_server, daemon=True)
        web_thread.start()
        
        return True

    def stop(self):
        logger.info("🛑 Остановка системы...")
        self.running = False
        
        if hasattr(self, 'pipe'):
            try:
                self.pipe.terminate()
                self.pipe.wait(timeout=5)
                logger.info("✅ FFmpeg остановлен")
            except Exception as e:
                logger.error(f"❌ Ошибка при остановке FFmpeg: {e}")
                try:
                    self.pipe.kill()
                except:
                    pass

def main():
    config = Config()
    processor = RTSPYOLOProcessor(config)
    
    try:
        if processor.start():
            logger.info("✅ Система запущена с УЛУЧШЕННЫМ трекингом")
            logger.info("🎯 Режим: ТОЛЬКО автомобили")
            logger.info("🛡️ Защита от дублирования пересечений: ВКЛЮЧЕНА")
            logger.info(f"⏰ Период охлаждения: {config.crossing_cooldown} кадров")
            logger.info("📏 Вертикальная линия пересечения")
            
            while True:
                time.sleep(1)
                
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