# BraTS2021 기반 뇌종양 영역 분할 프로젝트

BraTS2021 MRI 데이터를 활용한 2D UNet 기반 뇌종양 영역 분할(segmentation) 프로젝트입니다.  
모델은 FLAIR, T1, T1CE, T2 네 가지 MRI modality를 입력으로 받아 종양 영역 mask를 예측합니다.

## 프로젝트 구성

- `model.py`: 2D UNet 모델 정의
- `data_loader.py`: BraTS2021 2D slice 데이터셋 및 DataLoader 구성
- `train.py`: 모델 학습 및 checkpoint 저장
- `evaluate.py`: Dice, IoU, Precision, Recall, F1-score, ROC Curve, Confusion Matrix 평가
- `show_slice.py`: 특정 환자의 MRI slice와 segmentation mask 시각화
- `grad_cam.py`: Grad-CAM 기반 설명 가능 AI 시각화
- `make_segmentation_presentation_assets.py`: 발표용 Dice/IoU 차트 및 mask 비교 이미지 생성
- `smoke_test.py`: 데이터 로딩, 모델 forward, loss 계산 테스트

## 데이터 구조

데이터셋 경로는 `BRATS2021_DATA_DIR` 환경변수로 지정하거나, `data/` 폴더 아래에 연결하면 됩니다.

예상되는 BraTS2021 환자 폴더 구조는 다음과 같습니다.

```text
BraTS2021_00000/
  BraTS2021_00000_flair.nii.gz
  BraTS2021_00000_t1.nii.gz
  BraTS2021_00000_t1ce.nii.gz
  BraTS2021_00000_t2.nii.gz
  BraTS2021_00000_seg.nii.gz
```

데이터 폴더는 용량이 크기 때문에 GitHub에는 업로드하지 않으며, `.gitignore`에서 제외했습니다.

## 설치 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 연결

환경변수를 사용하는 방법:

```bash
export BRATS2021_DATA_DIR=/path/to/BraTS2021_Training_Data
```

또는 제공된 스크립트로 심볼릭 링크를 만들 수 있습니다.

```bash
python link_brats_data.py /path/to/BraTS2021_Training_Data
```

## 실행 방법

데이터 구조 확인:

```bash
python check_data.py
```

빠른 smoke test:

```bash
python smoke_test.py
```

모델 학습:

```bash
python train.py
```

성능 평가:

```bash
python evaluate.py
```

MRI slice 시각화:

```bash
python show_slice.py
```

발표용 결과 이미지 생성:

```bash
python make_segmentation_presentation_assets.py
```

## 주요 평가 지표

Brain Tumor Segmentation에서는 단순 Accuracy보다 예측 mask와 실제 mask의 겹침 정도가 중요합니다.  
따라서 본 프로젝트에서는 다음 지표를 중심으로 평가합니다.

- **Dice Score**: 예측한 종양 영역과 실제 종양 영역이 얼마나 겹치는지 나타내는 대표적인 segmentation 지표
- **IoU (Intersection over Union)**: 예측 영역과 실제 영역의 교집합을 합집합으로 나눈 값으로, Dice보다 더 엄격한 overlap 지표
- **Visual Mask Comparison**: Original MRI, Ground Truth Mask, Predicted Mask를 비교하여 모델의 분할 결과를 시각적으로 확인

## 발표용 결과 요약 예시

종양 전체 영역(binary segmentation) 기준으로 모델은 다음과 같은 성능을 보였습니다.

- Mean Dice Score: `0.855`
- Mean IoU: `0.762`

이는 예측한 종양 mask가 실제 종양 mask와 높은 수준으로 겹쳤다는 것을 의미합니다.  
다만 세부 종양 class 구분보다는 전체 종양 영역 탐지 기준에서 더 안정적인 성능을 보입니다.
