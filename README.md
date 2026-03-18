Semantic-Forest-lab

SMILES 기반 분자 구조를 **데이터셋별 도메인 온톨로지(ChEBI/DTO/BAO/GO/MeSH, optional: PATO/Thesaurus/SIO/BERO/CHEMINF/OCE/DINTO/DRON/OGCO/CHMO/Chem2Bio2RDF/OntoTox)** 로 변환한 뒤,
설명가능한 결정트리를 **개미군집 최적화(ACO) 기반 배깅(bootstrap aggregating)** 으로 학습해 분류 성능을 높이는 실험 레포입니다.

이 레포는 알고리즘별 버전을 **하나의 프로젝트 안에서** 관리합니다:

- **ID3**: Information Gain (`information_gain`)
- **C4.5**: Gain Ratio (`gain_ratio`)
- **C5.0**: Gain Ratio (`gain_ratio`, C4.5 계열 고도화)
- **CART**: Gini impurity (`gini`)
- **CHAID**: Chi-square (`chi_square`)
- **PIG**: Penalized Information Gain (`pig`) — 온톨로지 계층 깊이 기반 IG 보정
- **Semantic Similarity**: Wu-Palmer 유사도 기반 분할 (`semantic_similarity`)
- **PIG+Semantic**: PIG와 유사도의 결합 (`pig_semantic`)

## 주요 특징

- **단일 레포 다중 알고리즘 버전 관리**: ID3/C4.5/CART/ACO 공존
- **알고리즘 프로파일 기반 실행**: `configs/algorithms/*.json`
- **온톨로지 기반**: 데이터셋별 도메인 온톨로지를 사용하는 의미론적 결정트리
- **ACO + Bagging**: 개미군집 최적화(ACO)로 규칙/경로 탐색 후 배깅 앙상블 구성
- **설명 가능성**: 의사결정 과정을 명확하게 추적 가능
- **Rule Analysis**: 규칙 추출, 커스텀 규칙 고정 주입(Fixed Rules), Rule 성능 비교 리포트 제공

단일 트리로 학습하는 버전은 별도 레포로 분리했습니다:

- https://github.com/thkim-01/Semantic-Decision-Tree

## 개발 환경

- Python: 3.8+
- 주요 의존성: `owlready2`, `rdkit`, `scikit-learn`, `pandas`, `numpy`
- 선택 의존성(가속): `torch` (CUDA 가능 환경에서 GPU 사용)

## 설치

```bash
pip install -r requirements.txt

# (선택) PyTorch 가속 사용 시
# pip install torch
```

## 데이터/온톨로지 다운로드

대용량 파일(`data/`, `ontology/`)은 GitHub 저장소에 포함하지 않고 외부 링크로 제공합니다.

- Ontology 다운로드: https://drive.google.com/drive/folders/1-Ff2iOEhcAB4VRGPaHCdSWSEHFrM4T8X?usp=drive_link
- Dataset 다운로드: https://drive.google.com/drive/folders/1Aw2UvH8cCp43BiHUGRidCcudNb30pMkw?usp=drive_link

> 2026-03 업데이트: 신규 온톨로지 파일 `sio.owl`, `bero.owl`, `cheminf.owl`, `oce-merged.owl`, `dinto_1.3.owl`, `dron.owl`, `ogco.owl`, `chmo.owl`, `chem2bio2rdf.owl`, `OntoTox.owl` 도 위 Ontology 링크에 포함되어 있습니다.

Ontology 폴더에는 최소 다음 파일이 포함되어야 합니다.

- `chebi.owl`, `go.owl`, `mesh.owl`, `bao_complete.owl`, `DTO.xrdf`
- `pato.owl`, `Thesaurus.owl` (요청하신 Theasaurus 파일은 `Thesaurus.owl` 표기)
- `sio.owl`, `bero.owl`, `cheminf.owl`, `oce-merged.owl`, `dinto_1.3.owl`, `dron.owl`
- `ogco.owl`, `chmo.owl`, `chem2bio2rdf.owl`, `OntoTox.owl`

다운로드 후 프로젝트 루트에 아래 구조로 배치하세요.

```text
ACO-Semantic-Forest/
	data/
		...
	ontology/
		...
```

## 빠른 시작 (Quick Start)

### 0. 알고리즘 버전 실행 (권장)

```bash
# ID3 버전 실행
python experiments/run_semantic_forest_lab.py --algorithm id3

# C4.5 버전 실행
python experiments/run_semantic_forest_lab.py --algorithm c45

# CART 버전 실행
python experiments/run_semantic_forest_lab.py --algorithm cart
python experiments/run_semantic_forest_lab.py --algorithm aco

# (선택) torch 백엔드 사용
python experiments/run_semantic_forest_lab.py --algorithm id3 --compute-backend torch --torch-device auto
```

> 참고: 현재 코드의 주요 병목 중 일부는 온톨로지 객체 순회/정제(refinement) 판정 로직입니다.
> PyTorch 가속은 impurity/entropy 계산을 우선 가속하며, 전체 파이프라인을 완전 GPU-only로 바꾸지는 않습니다.

### 1. 단일 데이터셋 테스트 (BBBP)

```bash
# BBBP 데이터셋으로 Semantic Forest 학습 및 평가
python experiments/verify_semantic_forest.py --split-criterion information_gain
```

### 2. 전체 벤치마크 실행

```bash
# 모든 데이터셋에 대해 성능 평가
python experiments/verify_semantic_forest_multi.py

# 결과는 output/semantic_forest_benchmark.csv 및
# output/semantic_forest_benchmark_avg.csv에 저장됩니다
```

### 2-1. 멀티태스크 전체 태스크 + 덮어쓰기 실행

```bash
# 멀티태스크(Tox21, SIDER)의 모든 task를 평가하고 결과 파일을 덮어쓰기
python experiments/verify_semantic_forest_multi.py --all-tasks --overwrite

# 알고리즘 런처에서 동일 옵션 사용 (권장)
python experiments/run_semantic_forest_lab.py --algorithm cart --all-tasks --overwrite
python experiments/run_semantic_forest_lab.py --algorithm aco --datasets bbbp,clintox
```

### 3. 커스텀 설정으로 실행

```bash
# 특정 데이터셋만 실행
python experiments/verify_semantic_forest_multi.py --datasets bbbp,clintox

# 트리 개수 및 깊이 조정
python experiments/verify_semantic_forest_multi.py --n-estimators 50 --max-depth 8

# 개미군집 최적화(ACO) 탐색 강도 조정
python experiments/verify_semantic_forest_multi.py --search-strategy heuristic --max-candidate-refinements 64 --candidate-fraction 0.35 --heuristic-probe-samples 128

# 분할 기준 변경
python experiments/verify_semantic_forest_multi.py --split-criterion information_gain
python experiments/verify_semantic_forest_multi.py --split-criterion gain_ratio
python experiments/verify_semantic_forest_multi.py --split-criterion gini

# PIG / Semantic Similarity 알고리즘
python experiments/run_full_benchmark.py --algorithm pig --pig-alpha 1.5
python experiments/run_full_benchmark.py --algorithm semantic_similarity
python experiments/run_full_benchmark.py --algorithm pig_semantic --pig-alpha 1.0 --semantic-weight 0.3

# 알고리즘 이름으로 실행 (권장)
python experiments/verify_semantic_forest_multi.py --algorithm id3
python experiments/verify_semantic_forest_multi.py --algorithm c45
python experiments/verify_semantic_forest_multi.py --algorithm cart
python experiments/verify_semantic_forest_multi.py --search-strategy aco

# 알고리즘 런처에서 특정 데이터셋만 실행
python experiments/run_semantic_forest_lab.py --algorithm c45 --datasets bbbp,clintox
```

### 4. 규칙(Rule) 분석 및 고정 기능 활용

데이터셋에서 모델이 추출한 강력한 규칙들을 파일로 엑스포트하고, 유의미한 규칙을 Seed로 고정(Fix) 및 주입하여 다음 실험의 성능 및 설명 가능성을 한층 높일 수 있습니다.

```bash
# 기본 벤치마크 실행 시 각 태스크마다 상위 5개의 규칙이 output/benchmark_rules/ 폴더에 자동 추출됩니다.
python experiments/run_full_benchmark.py

# 파라미터로 추출 갯수와 포맷(json, csv, md 등)을 변경할 수 있습니다.
python experiments/run_full_benchmark.py --export-format csv --top-k 10

# 추출된 특정 JSON 규칙 파일(예: bbbp_p_np.json)을 별도의 폴더(예: my_fixed_rules/)에 복사/수정한 뒤 이를 고정 Seed로 주입하여 벤치마크를 재실행합니다.
python experiments/run_full_benchmark.py --fixed-rules-dir output/my_fixed_rules/
```

> **규칙 비교 리포트 기능**: `--fixed-rules-dir` 옵션을 사용하여 고정 규칙을 주입하고 실험이 실행된 경우, 해당 데이터셋의 `--export-rules-dir`(기본 `output/benchmark_rules/`) 쪽에 `_comparison.md` 파일명으로 마크다운 표 리포트가 자동 생성됩니다. 
이를 통해 규칙 고정/주입에 따른 학습 성능(AUC, F1, Accuracy, MAE 등)의 증감 차이를 명확하게 모니터링할 수 있습니다.

## 버전 구조

```text
configs/algorithms/
	id3.json
	c45.json
	cart.json
src/algorithms/
	profiles.py
experiments/
	run_semantic_forest_lab.py
```

## 알고리즘 설명

### ID3 (Iterative Dichotomiser 3)

- **분할 기준**: Information Gain을 사용하여 최적의 분할 지점 선택
- **Entropy**: $H(S) = -\sum_{i=1}^{n} p_i \log_2 p_i$
- **Information Gain**: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$
- **특징**: 해석이 직관적이며 엔트로피 기반으로 불확실성 감소를 최대화

### C4.5

- **분할 기준**: Gain Ratio
- **핵심 아이디어**: Information Gain을 split info로 정규화

### C5.0

- **분할 기준**: Gain Ratio (`gain_ratio`)
- **핵심 아이디어**: C4.5의 Gain Ratio 기반 분할을 계승한 실험 프로파일
- **구현 매핑**: 러너 옵션 `--algorithm c5.0` / `--algorithm c50` → `gain_ratio`

### CART

- **분할 기준**: Gini impurity
- **Gini impurity**: $Gini = 1 - \sum_{i=1}^{n} p_i^2$

### CHAID

- **분할 기준**: Chi-square (`chi_square`)
- **핵심 아이디어**: 분할 전/후 클래스 분포 차이를 카이제곱 통계량으로 평가
- **구현 매핑**: 러너 옵션 `--algorithm chaid` → `chi_square`

### PIG (Penalized Information Gain)

- **분할 기준**: 온톨로지 계층 깊이 기반 Taxonomic Informativeness로 Information Gain을 보정
- **핵심 수식**:
  - $IC(c) = -\log_2 P(c)$, where $P(c) = \frac{|\text{descendants}(c)|}{|\text{total concepts}|}$
  - $TI(c) = IC(c) \times \frac{\text{depth}(c)}{\text{max\_depth}}$
  - $ATI = \frac{1}{|C|} \sum_{c \in C} TI(c)$ (Average Taxonomic Informativeness)
  - $PIG(S) = IG(S) \times (1 + \log(1 + \alpha \times ATI))$
- **하이퍼파라미터**: `--pig-alpha` (기본값 1.0) — ATI 가중치
- **구현 매핑**: `--algorithm pig` → `pig`
- **특징**: 온톨로지 계층에서 더 구체적(깊은) 개념을 사용하는 분할에 높은 점수 부여

### Semantic Similarity (의미적 유사도 기반 분할)

- **분할 기준**: Wu-Palmer 유사도를 이용한 온톨로지 거리 기반 분할 점수
- **핵심 수식**:
  - $Sim_{WP}(a, b) = \frac{2 \times \text{depth}(LCA(a,b))}{\text{depth}(a) + \text{depth}(b)}$
  - $Sim(A) = \sum_u p_u \times Sim_{avg}(a_u)$ (파티션별 가중 유사도 합)
- **구현 매핑**: `--algorithm semantic_similarity` / `--algorithm semantic_sim` → `semantic_similarity`
- **특징**: 같은 파티션에 속한 인스턴스들이 온톨로지적으로 유사한 분할을 선호

### PIG + Semantic (결합)

- **분할 기준**: PIG와 Semantic Similarity의 가중 결합
- **핵심 수식**: $\text{Score} = (1 - w) \times PIG + w \times Sim(A) \times IG_{\max}$
- **하이퍼파라미터**: `--pig-alpha` (기본값 1.0), `--semantic-weight` (기본값 0.3, 범위 0~1)
- **구현 매핑**: `--algorithm pig_semantic` → `pig_semantic`
- **특징**: 두 온톨로지 신호를 동시에 활용하여 분할 품질 극대화

### 처리 과정

1. **입력**: `data/` 디렉토리의 CSV 파일 (예: `data/bbbp/BBBP.csv`)
2. **전처리**: SMILES → RDKit을 통한 분자 피처 추출
3. **온톨로지 변환**: 데이터셋별 1:1 도메인 온톨로지 매핑 기반 그래프 생성
4. **학습**: 개미군집 최적화(ACO) 기반 탐색 + 선택한 분할 기준(ID3/C4.5/C5.0/CART/CHAID/PIG/Semantic Similarity)으로 여러 트리 학습 (배깅)
5. **평가**: AUC-ROC, Accuracy 등 성능 지표 계산

## 데이터셋

지원하는 데이터셋:

- BBBP (Blood-Brain Barrier Penetration)
- BACE (β-secretase inhibitors)
- ClinTox (Clinical toxicity)
- HIV (HIV replication inhibition)
- Tox21 (Toxicity prediction)
- SIDER (Side effects)
- 기타 분자 특성 예측 데이터셋

### 데이터셋 ↔ 온톨로지 대응 (strict 1:1)

| Dataset | Ontology | Bridge Domain |
| ------- | -------- | ------------- |
| BBBP    | ChEBI    | chebi         |
| BACE    | DTO      | anchor        |
| ClinTox | ChEBI    | chebi         |
| HIV     | BAO      | anchor        |
| Tox21   | GO       | anchor        |
| SIDER   | MeSH     | anchor        |

> 기준 설정: `configs/dataset_ontology_config.json`, 라우팅 정책: `src/aco/ontology_router.py`

## 출력 결과

- **콘솔 로그**: 학습 진행 상황 및 성능 지표
- **CSV 파일(직접 실행)**: `output/semantic_forest_benchmark.csv` - 전체/부분 벤치마크 결과
- **CSV 파일(평균)**: `output/semantic_forest_benchmark_avg.csv` - 데이터셋별 task macro 평균
- **CSV 파일(알고리즘 런처)**: `output/lab_runs/semantic_forest_benchmark_{id3|c45|cart|aco}.csv`
- **Feature 캐시**: `output/feature_cache/*.sqlite3` (기본값, `--feature-cache-dir`로 변경 가능)
- **요약 파일**: 개별 실험 결과 요약

## 자주 쓰는 옵션

- `--all-tasks`: 멀티태스크 데이터셋(Tox21, SIDER)의 모든 task를 평가
- `--overwrite`: 기존 결과 CSV가 있으면 삭제 후 처음부터 다시 실행
- `--datasets bbbp,clintox`: 지정한 데이터셋만 실행
- `--algorithm id3|c45|cart|aco|pig|semantic_similarity|pig_semantic`: 알고리즘 프로파일 적용
- `--pig-alpha 1.0`: PIG 알고리즘의 ATI 가중치 (기본값 1.0)
- `--semantic-weight 0.3`: pig_semantic 결합 비중 (기본값 0.3, 범위 0~1)
- `--search-strategy exhaustive|aco`: 노드 분할 후보 탐색 전략 선택
- `--search-strategy heuristic|exhaustive`: 분할 후보 탐색 방식 선택
- `--max-candidate-refinements`: 휴리스틱 모드에서 노드당 평가할 후보 상한
- `--candidate-fraction`: 생성된 후보 중 정밀평가에 남길 비율
- `--heuristic-probe-samples`: 후보 사전순위 계산에 사용할 샘플 수

## 라이센스

본 프로젝트는 연구 및 교육 목적으로 사용 가능합니다.
