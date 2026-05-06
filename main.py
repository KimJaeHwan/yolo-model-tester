import os
import queue
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from capture import CaptureModule
from config import Config
from inference import InferenceModule
from logger import LoggingModule
from overlay import OverlayWindow
from smoother import DetectionSmoother


def main() -> None:
    config = Config()
    _check_model_files(config)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    capture = CaptureModule(config)
    inference = InferenceModule(config)
    logger = LoggingModule(config)

    param_path, bin_path = _model_paths(config, config.active_model)
    inference.load_model(param_path, bin_path)

    overlay = OverlayWindow(config)
    overlay.show()

    result_queue: queue.Queue = queue.Queue(maxsize=5)
    running = threading.Event()
    running.set()
    logging_enabled = threading.Event()
    frame_id_counter = [0]
    model_switch_lock = threading.Lock()

    smoother = DetectionSmoother(
        confirm_frames=config.confirm_frames,
        miss_frames=config.miss_frames,
        iou_threshold=config.iou_threshold,
    ) if config.smoothing_enabled else None

    # ── 추론 루프 ─────────────────────────────────────────────────────────────
    # 캡처 범위를 매 프레임 overlay 창 위치로 동기화

    def inference_loop() -> None:
        while running.is_set():
            # overlay 창 현재 위치를 캡처 범위로 설정
            x, y, w, h = overlay.get_scan_rect()
            if w <= 0 or h <= 0:
                continue
            capture.set_capture_rect(x, y, w, h)

            frame = capture.get_frame()
            if frame is None:
                continue

            # 크기 불일치 프레임 스킵 (rect 변경 직후 stale 프레임)
            fh, fw = frame.shape[:2]
            if fw != w or fh != h:
                continue

            try:
                detections = inference.infer(frame)
            except Exception as e:
                print(f"[Main] 추론 예외 (스킵): {e}")
                continue

            if smoother is not None:
                detections = smoother.update(detections)

            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                result_queue.put_nowait((detections, frame))
            except queue.Full:
                pass

    # ── Qt 타이머: 오버레이 갱신 ──────────────────────────────────────────────

    def update_ui() -> None:
        if result_queue.empty():
            return
        try:
            detections, frame = result_queue.get_nowait()
        except queue.Empty:
            return

        frame_id_counter[0] += 1
        fid = frame_id_counter[0]

        try:
            overlay.update_detections(detections)
        except Exception as e:
            print(f"[Main] UI 오류 (스킵): {e}")

        if logging_enabled.is_set():
            try:
                logger.log(detections, frame, fid)
            except Exception as e:
                print(f"[Main] 로거 오류: {e}")

    timer = QTimer()
    timer.timeout.connect(update_ui)
    timer.start(33)

    # ── 단축키 콜백 ───────────────────────────────────────────────────────────

    def toggle_logging() -> None:
        if logging_enabled.is_set():
            logging_enabled.clear()
            logger.stop_session()
            print("[Main] 로깅 중지")
        else:
            logging_enabled.set()
            logger.start_session()
            print("[Main] 로깅 시작")

    def manual_screenshot() -> None:
        frame = capture.get_frame()
        if frame is None:
            print("[Main] 스크린샷 실패: 프레임 없음")
            return
        detections = inference.infer(frame)
        path = logger.save_screenshot(frame, detections, frame_id_counter[0])
        print(f"[Main] 수동 스크린샷: {path}")

    def next_model() -> None:
        with model_switch_lock:
            cycle = config.model_cycle
            try:
                idx = cycle.index(config.active_model)
                next_idx = (idx + 1) % len(cycle)
            except ValueError:
                next_idx = 0
            next_name = cycle[next_idx]

            param, bin_ = _model_paths(config, next_name)
            if not os.path.exists(param) or not os.path.exists(bin_):
                print(f"[Main] 모델 없음 (건너뜀): {next_name}")
                return

            print(f"[Main] 모델 전환: {config.active_model} → {next_name}")
            try:
                inference.switch_model(next_name)
                config.active_model = next_name
                if smoother is not None:
                    smoother.reset()
                overlay.show_model_label(next_name)
            except Exception as e:
                print(f"[Main] 모델 전환 실패: {e}")

    def on_exit() -> None:
        print("[Main] 종료 중...")
        running.clear()
        timer.stop()
        capture.stop()
        if logging_enabled.is_set():
            logger.stop_session()
        app.quit()

    overlay.on_toggle_logging = toggle_logging
    overlay.on_manual_screenshot = manual_screenshot
    overlay.on_next_model = next_model
    overlay.on_exit = on_exit

    # ── 시작 ──────────────────────────────────────────────────────────────────

    print("[Main] 시작")
    print("[Main] 편집모드: 창 드래그/리사이즈로 탐지 범위 설정 → F7로 잠금")
    print("[Main] F2:로깅  F3:스크린샷  F4/F5:conf  F6:모델전환  ESC:종료")

    # 캡처 시작 전 overlay 초기 위치로 범위 선설정 (풀스크린 프레임 방지)
    app.processEvents()
    ix, iy, iw, ih = overlay.get_scan_rect()
    capture.set_capture_rect(ix, iy, iw, ih)

    capture.start()
    threading.Thread(target=inference_loop, daemon=True).start()

    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        on_exit()
        exit_code = 0

    sys.exit(exit_code)


def _model_paths(config: Config, prefix: str) -> tuple[str, str]:
    return (
        os.path.join(config.model_dir, f"{prefix}.param"),
        os.path.join(config.model_dir, f"{prefix}.bin"),
    )


def _check_model_files(config: Config) -> None:
    if not os.path.isdir(config.model_dir):
        print(f"[Error] models/ 폴더 없음: {os.path.abspath(config.model_dir)}")
        sys.exit(1)
    param, bin_ = _model_paths(config, config.active_model)
    if not os.path.exists(param) or not os.path.exists(bin_):
        print(f"[Error] 모델 파일 없음: {param}")
        sys.exit(1)


if __name__ == "__main__":
    main()
