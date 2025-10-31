# CNN 기반 감정 인식 학습 가이드

## 개요

HEAL 시스템은 이제 두 가지 감정 인식 방식을 지원합니다:

1. **AU 기반 방식 (간접)**: 얼굴 랜드마크 → AU 추출 → 감정 분류
   - 프라이버시 보호
   - 적은 학습 데이터로 가능
   - 해석 가능성 높음

2. **CNN 기반 방식 (직접)**: 얼굴 이미지 → CNN → 감정 분류
   - 높은 정확도 가능
   - 많은 학습 데이터 필요
   - End-to-end 학습

3. **앙상블 방식**: 두 방식의 예측을 가중 평균으로 결합

---

## 1단계: 얼굴 이미지 수집

### 방법 1: 비디오에서 얼굴 추출

```bash
python -m src.utils.label_faces --extract path/to/video.mp4 --input_dir data/faces_to_label
```

이 명령은 비디오에서 얼굴을 자동으로 추출하여 `data/faces_to_label/` 디렉토리에 저장합니다.

### 방법 2: 기존 얼굴 이미지 사용

얼굴 이미지를 직접 `data/faces_to_label/` 디렉토리에 복사합니다.

---

## 2단계: 얼굴 이미지 라벨링

대화형 라벨링 도구를 실행합니다:

```bash
python -m src.utils.label_faces --input_dir data/faces_to_label --output_dir data/labeled_faces
```

### 사용 방법:

- **[N] 또는 [0]**: Neutral로 라벨링
- **[H] 또는 [1]**: Happy로 라벨링
- **[S] 또는 [2]**: Sad로 라벨링
- **[U]**: 실행 취소 (이전 이미지로)
- **[Q]**: 종료 및 저장

라벨링된 이미지는 다음 구조로 저장됩니다:

```
data/labeled_faces/
    Neutral/
        face_0001.jpg
        face_0002.jpg
        ...
    Happy/
        face_0001.jpg
        face_0002.jpg
        ...
    Sad/
        face_0001.jpg
        face_0002.jpg
        ...
    labels.json
```

### 권장 사항:

- 각 감정당 최소 100개 이상의 이미지 수집
- 다양한 조명, 각도, 얼굴 표정 포함
- 클래스 불균형 최소화 (각 감정의 샘플 수를 비슷하게)

---

## 3단계: CNN 모델 학습

라벨링이 완료되면 CNN 모델을 학습합니다:

```bash
python -m src.models.train_cnn --data_dir data/labeled_faces --model_type mobilenet_v2 --num_epochs 20 --batch_size 32 --save_path models/cnn_emotion.pth
```

### 파라미터:

- `--data_dir`: 라벨링된 얼굴 이미지 디렉토리
- `--model_type`: CNN 아키텍처 (`mobilenet_v2`, `resnet18`, `efficientnet_b0`)
- `--num_epochs`: 학습 에폭 수 (기본값: 20)
- `--batch_size`: 배치 크기 (기본값: 32)
- `--learning_rate`: 학습률 (기본값: 0.001)
- `--val_split`: 검증 데이터 비율 (기본값: 0.2)
- `--save_path`: 학습된 모델 저장 경로

### 모델 선택 가이드:

- **mobilenet_v2**: 가볍고 빠름, 실시간 처리에 적합
- **resnet18**: 중간 크기, 균형 잡힌 성능
- **efficientnet_b0**: 높은 정확도, 더 많은 연산 필요

---

## 4단계: config.yaml 설정

학습이 완료되면 `config.yaml`에서 CNN 분류기를 활성화합니다:

### AU 기반만 사용 (기본값):

```yaml
emotion_classification:
  use_ml_classifier: false
  use_cnn_classifier: false
  use_ensemble: false
```

### CNN 기반만 사용:

```yaml
emotion_classification:
  use_cnn_classifier: true
  cnn_model_path: "models/cnn_emotion.pth"
  cnn_model_type: "mobilenet_v2"
  cnn_input_size: 224
```

### 앙상블 사용 (AU + CNN):

```yaml
emotion_classification:
  use_ensemble: true
  ensemble_weights:
    au_based: 0.6    # AU 기반 가중치
    cnn_based: 0.4   # CNN 기반 가중치
  cnn_model_path: "models/cnn_emotion.pth"
  cnn_model_type: "mobilenet_v2"
  cnn_input_size: 224
```

---

## 5단계: 시스템 실행

설정이 완료되면 일반적인 방식으로 시스템을 실행합니다:

```bash
python main.py
```

시스템은 자동으로 설정에 따라 적절한 분류기를 로드하고 사용합니다.

---

## 성능 비교

| 방식 | 장점 | 단점 | 권장 사용처 |
|------|------|------|-------------|
| **AU 기반 (규칙)** | - 학습 불필요<br>- 프라이버시 보호<br>- 해석 가능 | - 정확도 제한<br>- 복잡한 감정 인식 어려움 | 프로토타입, 프라이버시 중요 |
| **AU 기반 (ML)** | - 적은 데이터로 학습<br>- 프라이버시 보호<br>- 어느정도 해석 가능 | - AU 추출 정확도 의존 | 일반적인 사용 |
| **CNN 기반** | - 높은 정확도<br>- End-to-end 학습 | - 많은 데이터 필요<br>- 프라이버시 우려<br>- 해석 어려움 | 정확도 최우선 |
| **앙상블** | - 두 방식의 장점 결합<br>- 견고함 | - 연산량 증가 | 최고 성능 필요 |

---

## 문제 해결

### PyTorch 설치 오류

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

(CUDA 버전에 맞게 수정)

### GPU 사용

시스템은 자동으로 GPU를 감지하고 사용합니다. CUDA가 설치되어 있으면:

```python
[INFO] Using device: cuda
```

CPU만 사용 가능하면:

```python
[INFO] Using device: cpu
```

### 학습 데이터 부족

최소 권장 데이터:
- 각 감정당 100개 이상
- 총 300개 이상 (3개 감정)

데이터가 부족하면 Data Augmentation이 자동으로 적용됩니다:
- 좌우 반전
- 회전 (±10도)
- 밝기/대비 조정

---

## 고급 사용법

### 사용자 정의 데이터셋

자신만의 얼굴 이미지 데이터셋을 사용할 수 있습니다:

1. 감정별 디렉토리 구조 생성
2. `train_cnn.py`의 `--data_dir` 파라미터 지정
3. 감정 라벨은 `config.yaml`의 `emotion_definitions`와 일치해야 함

### Transfer Learning 활용

학습된 모델을 다른 감정으로 확장:

```python
from src.models.cnn_emotion_classifier import CNNEmotionClassifier

# 기존 모델 로드
classifier = CNNEmotionClassifier(model_path="models/cnn_emotion.pth")

# 추가 학습
classifier.train_model(train_loader, val_loader, num_epochs=10)
```

---

## 참고 자료

- PyTorch 공식 문서: https://pytorch.org/docs/
- Transfer Learning 가이드: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- FACS (Facial Action Coding System): Ekman & Friesen (1978)
