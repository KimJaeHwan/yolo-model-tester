import csv
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from config import Config
from inference import Detection


CLASS_COLORS_BGR = [
    (0, 0, 255),      # 0: 빨강
    (0, 255, 0),      # 1: 초록
    (255, 0, 0),      # 2: 파랑
    (0, 255, 255),    # 3: 노랑
    (255, 0, 255),    # 4: 마젠타
    (255, 255, 0),    # 5: 시안
    (0, 128, 255),    # 6: 주황
    (255, 0, 128),    # 7: 보라
]


class LoggingModule:
    def __init__(self, config: Config):
        self.config = config
        self._session_dir: str = ""
        self._screenshot_dir: str = ""
        self._csv_path: str = ""
        self._csv_file = None
        self._csv_writer = None
        self._lock = threading.Lock()
        self._active = False

        self._start_time: float = 0.0
        self._total_frames: int = 0
        self._total_detections: int = 0
        self._total_screenshots: int = 0

    def start_session(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.config.log_dir, ts)
        screenshot_dir = os.path.join(session_dir, "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        csv_path = os.path.join(session_dir, "detections.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "frame_id", "class_id",
                         "x1", "y1", "x2", "y2", "confidence", "screenshot"])

        with self._lock:
            self._session_dir = session_dir
            self._screenshot_dir = screenshot_dir
            self._csv_path = csv_path
            self._csv_file = csv_file
            self._csv_writer = writer
            self._active = True
            self._start_time = time.time()
            self._total_frames = 0
            self._total_detections = 0
            self._total_screenshots = 0

        print(f"[Logger] 세션 시작 → {session_dir}")
        return session_dir

    def log(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        frame_id: int,
    ) -> None:
        with self._lock:
            if not self._active or self._csv_writer is None:
                return
            self._total_frames += 1
            self._total_detections += len(detections)

        ts = datetime.now().isoformat(timespec="milliseconds")
        screenshot_path = ""

        # conf 초과 탐지 있으면 스크린샷 저장
        if self.config.auto_screenshot and any(
            d.confidence >= self.config.screenshot_conf_threshold for d in detections
        ):
            screenshot_path = self.save_screenshot(frame, detections, frame_id)

        with self._lock:
            if self._csv_writer is None:
                return
            try:
                for det in detections:
                    self._csv_writer.writerow([
                        ts, frame_id,
                        det.class_id,
                        det.x1, det.y1, det.x2, det.y2,
                        f"{det.confidence:.4f}",
                        screenshot_path,
                    ])
                # 버퍼 플러시 (성능 vs 내구성 트레이드오프: 10프레임마다)
                if self._total_frames % 10 == 0:
                    self._csv_file.flush()
            except Exception as e:
                print(f"[Logger] CSV 기록 오류: {e}")

    def save_screenshot(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_id: int,
    ) -> str:
        ts = datetime.now().strftime("%H%M%S%f")[:10]
        filename = f"frame_{frame_id:06d}_{ts}.png"
        path = os.path.join(self._screenshot_dir, filename)

        annotated = frame.copy()
        for det in detections:
            color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"cls{det.class_id}: {det.confidence:.2f}"
            cv2.putText(annotated, label, (det.x1 + 2, det.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 비동기 저장
        t = threading.Thread(
            target=self._write_image,
            args=(path, annotated),
            daemon=True,
        )
        t.start()

        with self._lock:
            self._total_screenshots += 1

        return os.path.join("screenshots", filename)

    def stop_session(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            elapsed = time.time() - self._start_time
            total_frames = self._total_frames
            total_detections = self._total_detections
            total_screenshots = self._total_screenshots
            session_dir = self._session_dir

            if self._csv_file:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                except Exception:
                    pass
                self._csv_file = None
                self._csv_writer = None

        # 요약 출력
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        fps = total_frames / elapsed if elapsed > 0 else 0.0

        print("\n=== 세션 종료 요약 ===")
        print(f"세션 시간: {h:02d}:{m:02d}:{s:02d}")
        print(f"총 프레임: {total_frames:,}")
        print(f"탐지 이벤트: {total_detections:,}")
        print(f"평균 FPS: {fps:.1f}")
        print(f"저장 스크린샷: {total_screenshots}장")
        print(f"로그 경로: {session_dir}/")
        print("=====================\n")

    @property
    def is_active(self) -> bool:
        return self._active

    def _write_image(self, path: str, img: np.ndarray) -> None:
        try:
            cv2.imwrite(path, img)
        except Exception as e:
            # 디스크 풀 등 오류 처리
            print(f"[Logger] 이미지 저장 실패 ({path}): {e}")
            with self._lock:
                if self._active:
                    self._active = False
                    print("[Logger] 경고: 디스크 용량 부족 — 로깅 자동 중지")
