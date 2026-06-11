# CardioCare — End-to-End Heart Disease ML System

**윤리 선언:** 본 시스템은 심장 전문의의 의사결정을 보조하는 도구이며, 독립적인 진단 자격이 없습니다. 모든 최종 의료 결정은 면허를 가진 의사가 내려야 합니다.

## 저장소 구조

```
.
├── data/                          # 데이터 또는 데이터 fetch 스크립트
│   ├── download_data.py
│   └── sample_batch.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── mlruns/                        # MLflow 산출물 (용량 과대 시 report.pdf 스크린샷)
├── Dockerfile
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── report.pdf
└── README.md
```

## 전체 재현 절차

채점자는 **README만 보고** 아래 순서로 전체 과정을 재현할 수 있어야 합니다.

```bash
# 1. clone
git clone https://github.com/touyoupo/cardiocare.git
cd cardiocare

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 학습 (데이터 없으면 data/download_data.py 자동 실행)
python src/train.py

# 4. Docker 이미지 빌드
docker build -t cardiocare:1.0 .

# 5. 단위 테스트
python -m unittest discover -s tests -v
```

### 선택 실행 (보고서·모니터링)

```bash
# Docker 컨테이너 추론
docker run --rm cardiocare:1.0

# 드리프트 모니터링
python src/monitor.py

# MLflow UI (mlruns/ 생성 후)
mlflow ui
```

## Dataset

- **Source:** [UCI Heart Disease — Cleveland subset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Script:** `data/download_data.py` (`train.py` 실행 시 자동 호출)
- **Target:** `0 = healthy`, `1 = heart disease`

## Design Highlights

- **No data leakage:** split first, fit preprocessors on train only
- **seed=42** everywhere; pinned `requirements.txt`
- **Model selection:** highest **recall** (reduce false negatives)
- **MLflow:** 3 model families logged; screenshots in `report.pdf`
- **CI:** https://github.com/touyoupo/cardiocare/actions

## Feature Store & Model Registry (§5.3)

| 항목 | 내용 | 선정 이유 |
|------|------|-----------|
| **Feature Store (1개)** | `chol` (콜레스테롤) | 심혈관 위험의 핵심 연속 지표; IQR 클리핑·표준화 후 재사용 가능 |
| **Model Registry (1개 메타데이터)** | `recall_on_holdout` | 위음성(FN) 위험 최소화를 위해 모델 승격·롤백의 1차 기준 |

## DVC (선택 확장)

`.dvc/`, `dvc.yaml`, `dvc.lock`, `params.yaml` — `dvc repro`로 파이프라인 재현 가능.

## AI Tool Disclosure

본 프로젝트는 ChatGPT를 코드 템플릿 작성 및 디버깅에 사용했으며, 모든 핵심 로직과 실험 결과는 본인이 독립적으로 완성했습니다. 자세한 내용은 `report.pdf` 부록 참조.
