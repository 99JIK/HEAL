# HEAL - Human Emotion Analysis using Landmarks

> FACS 기반 실시간 감정 인식 시스템

## 개요

HEAL은 FACS(Facial Action Coding System) 기반의 비침습적 감정 인식 시스템입니다. MediaPipe를 활용하여 얼굴 랜드마크를 추출하고, Action Unit(AU) 분석을 통해 감정 상태를 실시간으로 분류합니다. 개인정보 보호를 위해 이미지 대신 정규화된 랜드마크 좌표만 저장하며, 시계열 데이터베이스와의 호환성을 제공합니다.

## 주요 특징

- **실시간 감정 인식**: 웹캠을 통한 실시간 얼굴 표정 분석
- **개인정보 보호**: 실제 이미지를 저장하지 않고 상대 좌표만 저장
- **다중 피험자 지원**: 여러 명의 얼굴을 동시에 추적 및 분석
- **FACS 기반**: 해부학적으로 검증된 AU 측정 방식 적용
- **확장 가능한 데이터 저장**: JSON, CSV, InfluxDB 라인 프로토콜 지원
- **ML 분류기 지원**: 규칙 기반 및 머신러닝 기반 감정 분류 모두 지원

## 시스템 아키텍처

```
입력 (웹캠) → 얼굴 검출 → 얼굴 추적 → AU 추출 → 감정 분류 → 데이터 저장
               (Blaze Face)  (Centroid)   (Face Mesh)  (Classifier)  (JSON/CSV)
```

## 감정 분류 체계

본 시스템은 다음 3가지 기본 감정 상태를 인식합니다:

| 감정        | Action Units     | 설명                                       |
| ----------- | ---------------- | ------------------------------------------ |
| **Happy**   | AU6 + AU12       | 뺨 올림 + 입꼬리 올림                      |
| **Sad**     | AU1 + AU4 + AU15 | 내측 눈썹 올림 + 눈썹 찡그림 + 입꼬리 내림 |
| **Neutral** | -                | AU 활성화 임계값 미달                      |

각 감정은 0-100% 강도로 정량화되며, 5프레임 이동평균을 통해 시간적 안정성을 확보합니다.

## 구현 단계

### 1단계: 얼굴 검출 및 추적

**기술 스택**

- MediaPipe Blaze Face (얼굴 검출)
- Centroid + IoU 기반 추적 알고리즘

**기능**

- 마스크 착용 시에도 얼굴 검출 가능
- 다중 피험자 동시 처리
- 고유 ID 기반 프레임 간 추적
- 1초 안정성 임계값 (30프레임 연속 검출 후 ID 부여)
- 원형 큐 ID 관리 (0-999999)

**관련 파일**

- `src/models/face_detector.py` - 얼굴 검출기
- `src/models/face_tracker.py` - 얼굴 추적기

### 2단계: Action Unit 추출

**기술 스택**

- MediaPipe Face Mesh (478개 랜드마크)
- 기하학적 거리 기반 AU 측정

**측정 AU**

- AU1 (Inner Brow Raiser): 내측 눈썹 올림
- AU4 (Brow Lowerer): 눈썹 찡그림
- AU6 (Cheek Raiser): 뺨 올림
- AU12 (Lip Corner Puller): 입꼬리 올림
- AU15 (Lip Corner Depressor): 입꼬리 내림

**데이터 관리**

- ID별 독립적인 시계열 데이터 저장
- 최대 60초 히스토리 보관
- 통계값 자동 계산 (평균, 표준편차, 최대/최소)

**관련 파일**

- `src/models/au_extractor.py` - AU 측정
- `src/models/au_storage.py` - AU 시계열 데이터 관리

### 3단계: 감정 분류

**분류 방식**

- 규칙 기반 분류기 (기본)
- 머신러닝 분류기 (선택 가능)
  - Random Forest
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)

**특징**

- 시간적 평활화 (5프레임 이동평균)
- 감정 강도 정량화 (0-100%)
- 설정 파일 기반 분류기 전환

**관련 파일**

- `src/models/emotion_classifier.py` - 규칙 기반 분류기
- `src/models/ml_emotion_classifier.py` - ML 분류기
- `src/utils/data_labeling_tool.py` - 데이터 라벨링 도구
- `src/train_emotion_model.py` - ML 모델 학습

### 4단계: 개인정보 보호 데이터 저장

**저장 포맷**

- **JSON**: 전체 478개 랜드마크 정규화 좌표 및 메타데이터
- **CSV**: AU 값과 감정 분류 결과 (시계열 분석용)
- **InfluxDB**: 라인 프로토콜 형식 (시계열 DB 직접 import)

**개인정보 보호**

- 실제 이미지를 저장하지 않음
- 얼굴 영역 내 상대 좌표로 정규화 (0-1 범위)
- 저장된 좌표로 랜드마크 시각화 재생성 가능

**샘플링 설정**

- `config.yaml`에서 저장 빈도 조절
- 권장값: sample_rate=5 (30fps → 6fps)

**관련 파일**

- `src/utils/landmark_storage.py` - 랜드마크 저장 관리
- `src/utils/landmark_viewer.py` - 저장된 데이터 재생 뷰어
- `view_landmarks.py` - 랜드마크 뷰어 실행 프로그램

### 5단계: 누적 데이터 분석 (진행중)

**예정 기능**

- 감정별 빈도 분석
- 지속시간 통계
- 강도 분포 분석
- 시계열 DB 연동을 통한 대규모 데이터 처리

## 설치 방법

### 요구사항

- Python 3.8 이상
- 웹캠 (실시간 처리 시)

### 패키지 설치

```bash
pip install -r requirements.txt
```

**주요 의존성**

- `mediapipe>=0.10.0` - 얼굴 검출 및 랜드마크 추출
- `opencv-python>=4.8.0` - 영상 처리
- `numpy>=1.24.0` - 수치 계산
- `pyyaml>=6.0` - 설정 파일 로더
- `scikit-learn>=1.3.0` - 머신러닝 분류기 (선택)

## 사용 방법

### 1. 실시간 감정 분석

```bash
python -m src.models.emotion_analyzer
```

**키보드 컨트롤**

- `q` - 종료
- `s` - 스크린샷 저장
- `c` - AU 데이터 초기화
- `l` - 랜드마크 표시 ON/OFF
- `m` - 랜드마크 모드 전환 (simple/full/mesh)
- `r` - 녹화 시작/중지

**랜드마크 모드**

- `simple`: AU 관련 랜드마크만 표시 (고성능)
- `full`: 478개 전체 랜드마크 표시
- `mesh`: 전체 랜드마크 + 연결선 표시 (OpenFace 스타일)

### 2. 저장된 데이터 재생

```bash
python view_landmarks.py
```

저장된 랜드마크 JSON 파일을 불러와 프레임별로 재생하고 검토할 수 있습니다.

**키보드 컨트롤**

- `←/→` - 이전/다음 프레임
- `↑/↓` - 다음/이전 프레임 (보조)
- `S` - 현재 프레임 이미지 저장
- `A` - 전체 프레임 이미지 일괄 저장
- `Q` - 종료

### 3. ML 분류기 학습

#### 3.1 데이터 라벨링

```bash
python -m src.utils.data_labeling_tool
```

녹화된 데이터에 감정 라벨을 할당합니다.

**키보드 컨트롤**

- `0` - Neutral
- `1` - Happy
- `2` - Sad
- `←/→` - 이전/다음 프레임
- `Q` - 저장 및 종료

#### 3.2 모델 학습

```bash
python -m src.train_emotion_model --model_type random_forest
```

라벨링된 데이터를 사용하여 ML 모델을 학습합니다.

#### 3.3 ML 분류기 활성화

`config.yaml` 수정:

```yaml
emotion_classification:
  use_ml_classifier: true
  model_path: "models/emotion_classifier.pkl"
  model_type: "random_forest"
```

### Python API

```python
from src.models.emotion_analyzer import EmotionAnalyzer

# 초기화
analyzer = EmotionAnalyzer(
    enable_tracking=True,
    show_landmarks=True,
    landmark_mode='simple'
)

# 프레임 분석
results = analyzer.analyze(frame)
# 반환값: {face_id: {'bbox': [...], 'aus': {...}, 'emotion': 'Happy', 'confidence': 0.85}}

# 통계 조회
stats = analyzer.get_statistics(face_id=0, window_seconds=5.0)

# 성능 메트릭
fps = analyzer.get_fps()
proc_time = analyzer.get_avg_processing_time()
```

## 설정 파일

`config.yaml`에서 시스템 동작을 조정할 수 있습니다.

### 얼굴 검출

```yaml
face_detection:
  model_selection: 1 # 0=단거리(2m), 1=장거리(5m)
  min_detection_confidence: 0.5
```

### 얼굴 추적

```yaml
face_tracking:
  enabled: true
  max_disappeared: 30 # ID 제거 전 최대 프레임
  iou_threshold: 0.3
  min_stable_frames: 30 # ID 부여 전 최소 검출 프레임
```

### AU 추출

```yaml
au_extraction:
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  target_aus:
    - au1
    - au4
    - au6
    - au12
    - au15
```

### 감정 분류

```yaml
emotion_classification:
  use_ml_classifier: false # ML 모델 사용 여부
  model_path: "models/emotion_classifier.pkl"
  model_type: "random_forest" # random_forest, svm, mlp
  smoothing_window: 5 # 시간적 평활화 윈도우
```

### 녹화

```yaml
recording:
  output_dir: "data/recordings"
  sample_rate: 10 # N프레임마다 저장
```

**sample_rate 권장값**

- `1`: 전체 프레임 (30fps → 30fps) - 고정밀도, 대용량
- `5`: 5프레임마다 (30fps → 6fps) - 권장
- `10`: 10프레임마다 (30fps → 3fps) - 저용량
- `30`: 30프레임마다 (30fps → 1fps) - 최소 용량

### 시각화

```yaml
visualization:
  show_landmarks: false
  landmark_mode: "simple" # simple, full, mesh
  show_au_values: true
  show_emotion_bar: true
```

## 프로젝트 구조

```
HEAL/
├── config.yaml                      # 시스템 설정 파일
├── requirements.txt                 # Python 의존성
├── README.md
├── view_landmarks.py               # 랜드마크 뷰어 실행 프로그램
│
├── src/
│   ├── models/
│   │   ├── face_detector.py        # 얼굴 검출 (Blaze Face)
│   │   ├── face_tracker.py         # 얼굴 추적 (Centroid + IoU)
│   │   ├── au_extractor.py         # AU 특징 추출 (Face Mesh)
│   │   ├── au_storage.py           # AU 시계열 데이터 관리
│   │   ├── emotion_classifier.py   # 규칙 기반 감정 분류기
│   │   ├── ml_emotion_classifier.py # ML 기반 감정 분류기
│   │   └── emotion_analyzer.py     # 통합 감정 분석 시스템
│   │
│   ├── utils/
│   │   ├── config_loader.py        # 설정 파일 로더
│   │   ├── landmark_storage.py     # 랜드마크 저장 (개인정보 보호)
│   │   ├── landmark_viewer.py      # 저장된 데이터 재생 뷰어
│   │   └── data_labeling_tool.py   # 데이터 라벨링 도구
│   │
│   └── train_emotion_model.py      # ML 모델 학습 스크립트
│
└── data/
    └── recordings/                  # 녹화 데이터 저장 디렉토리
        ├── landmarks_*.json         # 랜드마크 좌표
        ├── timeseries_*.csv         # AU 시계열 데이터
        ├── influxdb_*.txt           # InfluxDB 라인 프로토콜
        └── labels_*.csv             # 라벨링된 데이터
```

## 기술 스택

| 구성요소      | 기술                                   | 상태 |
| ------------- | -------------------------------------- | ---- |
| 얼굴 검출     | MediaPipe Blaze Face                   | ✅   |
| 얼굴 추적     | Centroid + IoU Tracking                | ✅   |
| AU 추출       | MediaPipe Face Mesh (478 landmarks)    | ✅   |
| AU 저장       | 시계열 데이터 + 통계                   | ✅   |
| 감정 분류     | 규칙 기반 + ML (Random Forest/SVM/MLP) | ✅   |
| 데이터 저장   | JSON/CSV/InfluxDB (개인정보 보호)      | ✅   |
| 데이터 라벨링 | 인터랙티브 라벨링 도구                 | ✅   |
| ML 학습       | 교차 검증 기반 학습 파이프라인         | ✅   |
| 누적 분석     | 통계 처리 및 시계열 분석               | 🔄   |

## 성능

- **얼굴 검출**: ~15-30 FPS (720p)
- **AU 추출**: ~20-30 FPS (simple 모드)
- **전체 파이프라인**: ~15-25 FPS (실시간 처리 가능)

_성능은 하드웨어 및 설정에 따라 달라질 수 있습니다._

## 개인정보 보호

본 시스템은 다음과 같은 개인정보 보호 조치를 적용합니다:

1. **이미지 미저장**: 실제 얼굴 이미지를 저장하지 않음
2. **좌표 정규화**: 얼굴 영역 내 상대 좌표(0-1)로 변환하여 저장
3. **재현 가능성**: 저장된 좌표로 랜드마크 시각화 재생성 가능
4. **익명성**: 개인 식별이 불가능한 기하학적 특징만 저장

## 참고 문헌

- Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System: A Technique for the Measurement of Facial Movement. Consulting Psychologists Press.
- Bazarevsky, V., et al. (2019). BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs. arXiv:1907.05047
- Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172

## 라이선스

본 프로젝트는 주식회사 와이디자인랩을 위해 개발되었습니다.

## 문의

기술적 문의사항이나 버그 리포트는 이슈 트래커를 통해 제출해 주시기 바랍니다.

---

**개발 상태**: 4단계 완료 (얼굴 검출 + 추적 + AU 추출 + 감정 분류 + 데이터 저장 + ML 학습)
