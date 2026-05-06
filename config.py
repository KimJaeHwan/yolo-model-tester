from dataclasses import dataclass, field


@dataclass
class Config:
    # 모델 설정
    model_dir: str = "models"
    active_model: str = "best.ncnn"
    input_size: int = 640              # 0이면 .param에서 자동 감지
    num_classes: int = 0               # 0이면 출력 채널에서 자동 계산

    # 추론 설정
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    num_threads: int = 4
    min_box_size: int = 10             # 이보다 작은 박스(px) 필터링

    # 캡처 설정
    capture_mode: str = "mss"          # "mss" | "dxcam" (mss가 다중 모니터 안전)
    target_window_title: str = ""      # 비어있으면 전체 화면
    capture_fps: int = 30
    monitor_index: int = 0             # 다중 모니터 시 인덱스

    # 오버레이 설정
    overlay_alpha: float = 0.85
    show_confidence: bool = True
    show_class_id: bool = True
    box_thickness: int = 2

    # 로깅 설정
    log_dir: str = "logs"
    auto_screenshot: bool = True
    screenshot_conf_threshold: float = 0.6

    # 스무딩 설정
    smoothing_enabled: bool = True
    confirm_frames: int = 2     # N프레임 연속 감지 후 표시
    miss_frames: int = 3        # N프레임 연속 미감지 후 제거
    iou_threshold: float = 0.3  # 같은 박스로 볼 IOU 기준

    # 뷰어 설정
    preview_mode: bool = True           # True: 리사이즈 가능한 뷰어 창 / False: 게임 위 투명 오버레이
    preview_width: int = 960
    preview_height: int = 540

    # 런타임 상태 (직접 수정 가능)
    model_cycle: list = field(default_factory=lambda: [
        "best.ncnn", "model1", "model2", "model3", "model4", "model5"
    ])
