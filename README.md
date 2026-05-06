# yolo-model-tester

분석 대상 애플리케이션에서 추출한 YOLO-NCNN 모델을 강제 구동하여,  
**실제로 무엇을 탐지하는지 실시간으로 검증하는 내부 분석 도구**입니다.

추출된 모델 파일(`.param` / `.bin`)을 `models/` 폴더에 배치하면  
해당 모델이 현재 화면에서 어떤 오브젝트를 탐지하는지 오버레이로 확인할 수 있습니다.

---

## 사용 목적

- 의심 매크로/치트 애플리케이션에서 추출한 YOLO 모델의 동작 검증
- 모델이 게임 화면 어느 요소에 반응하는지 시각적으로 확인
- 여러 모델을 런타임에 전환하며 각각의 탐지 대상 비교 분석

> **주의**: 이 도구는 추출된 모델의 기능 분석 목적으로만 사용합니다.

---

## 요구사항

- Windows 10/11 (64-bit)
- Python 3.10+

---

## 설치

```bash
# 1. 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt
```

> **ncnn**은 별도 설치가 필요한 경우  
> [ncnn Releases](https://github.com/Tencent/ncnn/releases) 에서 Python wheel 다운로드 후:
> ```bash
> pip install ncnn-*.whl
> ```

---

## 모델 파일 배치

분석 대상 앱에서 추출한 YOLO-NCNN 모델 파일을 `models/` 폴더에 넣습니다.

```
models/
├── best.ncnn.param      ← 추출된 모델
├── best.ncnn.bin
├── model1.param         ← 추가 모델 (선택)
├── model1.bin
└── ... (model2 ~ model5)
```

지원 모델 구조:
- **YOLOv5** (앵커 기반): 출력 채널 수로 자동 감지
- **YOLOv8** (앵커-프리): `.param` 내 Reshape 레이어로 자동 감지

---

## 실행

```bash
cd yolo_overlay   # 또는 프로젝트 루트
python main.py
```

실행 시 파란 테두리의 스캔 영역 창이 나타납니다.

---

## 사용 방법

### 1단계 — 탐지 범위 설정

| 동작 | 방법 |
|------|------|
| 창 이동 | 상단 파란 바 드래그 |
| 창 크기 조절 | 모서리/가장자리 핸들 드래그 |

분석하려는 게임/앱 화면 위에 창을 위치시킵니다.

### 2단계 — 오버레이 모드 전환

**F7** 을 누르면 클릭 통과 모드로 전환됩니다.  
이 상태에서 탐지 박스가 화면 위에 실시간으로 표시되며,  
마우스 클릭은 아래의 게임/앱으로 그대로 전달됩니다.

### 단축키

| 키 | 동작 |
|----|------|
| F7 | 오버레이 잠금 / 편집 모드 전환 |
| F2 | 로깅 시작 / 중지 (CSV + 스크린샷 저장) |
| F3 | 수동 스크린샷 저장 |
| F4 | 탐지 임계값(conf) +0.05 |
| F5 | 탐지 임계값(conf) -0.05 |
| F6 | 다음 모델로 전환 (순환) |
| ESC | 종료 |

---

## 분석 팁

### 모델별 탐지 대상 파악
F6으로 모델을 순환하며 어떤 오브젝트에 박스가 붙는지 비교합니다.

```
best.ncnn  → 8개 클래스 복합 탐지 (다양한 요소 동시 인식)
model1~4   → 단일 클래스 (특정 대상 하나에 집중)
model5     → YOLOv8 구조, 정밀도 높음
```

### 로그 분석
F2로 로깅 시작 후 `logs/세션폴더/detections.csv` 분석:

```python
import pandas as pd

df = pd.read_csv("logs/세션폴더/detections.csv")
print(df['class_id'].value_counts())   # 클래스별 탐지 빈도
print(df['confidence'].describe())     # 신뢰도 분포
```

### 탐지가 너무 많을 때
F4로 conf 임계값을 올려 고신뢰도 탐지만 표시합니다.  
권장 범위: `0.6 ~ 0.8` (비게임 화면에서 오탐 감소)

---

## 프로젝트 구조

```
yolo_overlay/
├── main.py          # 진입점, 파이프라인 조합
├── config.py        # 설정 (conf, 캡처 모드, 스무딩 등)
├── capture.py       # 화면 캡처 (mss 기본, 다중 모니터 지원)
├── inference.py     # NCNN 추론, YOLOv5/v8 자동 분기
├── overlay.py       # 투명 오버레이 창 (드래그/리사이즈)
├── smoother.py      # IOU 기반 탐지 안정화
├── logger.py        # CSV + PNG 비동기 로깅
├── viewer.py        # 프리뷰 뷰어 (참고용)
├── models/          # 추출된 모델 파일 위치 (.param/.bin)
├── logs/            # 세션별 탐지 로그 자동 생성
└── requirements.txt
```

---

## 빌드 (단일 EXE)

```bash
pyinstaller build.spec
```
