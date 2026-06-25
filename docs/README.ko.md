# SimXRD-4M [ICLR 2025](https://iclr.cc/virtual/2025/poster/28452)

<p align="center">
  <strong>언어:</strong>
  <a href="../README.md">English</a> | <a href="README.zh-CN.md">中文</a> | <a href="README.ja.md">日本語</a> | 한국어 | <a href="README.de.md">Deutsch</a> | <a href="README.es.md">Español</a>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=mkuB677eMM">
    <img src="https://img.shields.io/badge/ICLR-2025%20OpenReview-4b44ce" />
  </a>
  <a href="https://openreview.net/forum?id=mkuB677eMM">
    <img src="https://img.shields.io/badge/Paper-OpenReview-4b44ce" />
  </a>
  <a href="https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E">
    <img src="https://img.shields.io/badge/Dataset-OneDrive-0078D4?logo=microsoft-onedrive&logoColor=white" />
  </a>
  <a href="https://github.com/compasszzn/XRDBench">
    <img src="https://img.shields.io/badge/Benchmark-Code-blue?logo=github" />
  </a>
  <a href="https://github.com/Bin-Cao/SimXRD/stargazers">
    <img src="https://img.shields.io/github/stars/Bin-Cao/SimXRD?logo=github&label=Stars" />
  </a>
  <a href="../LICENSE">
    <img src="https://img.shields.io/github/license/Bin-Cao/SimXRD" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" />
  </a>
  <a href="https://pypi.org/project/Pysimxrd/">
    <img src="https://img.shields.io/badge/PyPI-Pysimxrd-3775A9?logo=pypi&logoColor=white" />
  </a>
</p>

> [!IMPORTANT]
> **SimXRD는 AI 기반 X선 회절 연구를 위한 최초의 진정한 대규모 파운데이션 데이터셋입니다.**  
> SimXRD 이전에는 결정학 분야의 머신러닝 발전이 충분히 크고, 물리적으로 현실적이며, 체계적으로 벤치마킹된 회절 데이터셋의 부재로 제한되었습니다. SimXRD는 119,000개 이상의 결정 구조와 33가지 물리적으로 다양한 실험 조건을 포괄하는 400만 개 이상의 고충실도 시뮬레이션 XRD 패턴을 제공하여 이러한 상황을 바꾸었습니다. 제한적이거나 이상화된 패턴을 포함하는 기존 회절 데이터베이스와 달리, SimXRD는 피크 브로드닝, 격자 섭동, 기기 효과, 대칭 보존 변환 등 실제 변동성을 명시적으로 모델링합니다. 또한 SimXRD는 회절 표현 학습을 위한 최초의 표준화된 대규모 벤치마크 중 하나를 확립하여 현대 AI 모델의 학습, 평가, 확장을 가능하게 했습니다. 이 작업은 [XQueryer](https://github.com/Bin-Cao/XQueryer) 및 [XDecomposer](https://github.com/Licht0812/XDecomposer) 같은 시스템의 데이터 기반이 되었습니다.

**데이터 설명:** 결정은 230개의 공간군으로 분류되며, 각 공간군은 서로 다른 대칭 범주를 나타냅니다. 결정 구조에 대응하는 XRD 패턴은 재료 연구의 핵심 도구입니다. 그러나 XRD 패턴은 측정 환경, 장비, X선 광원, 입자 크기와 배향 같은 시료 특성의 영향을 받습니다. 따라서 강도 변화와 피크 브로드닝 등 다양한 특징이 나타나며, 정확한 상 식별이 어려워질 수 있습니다. 본 데이터베이스는 다양한 환경 조건에서의 회절 스펙트럼을 제공하여 모델 학습을 지원하고, 궁극적으로 XRD 패턴에서 올바른 공간군을 정확히 식별하는 것을 목표로 합니다.

---

## 설치

데이터베이스 파일을 처리하려면 다음 라이브러리가 필요합니다.

- ase
- tqdm

```bash
pip install ase tqdm
```

Kaggle 대회 안내

고급 모델을 벤치마킹하고 발전을 촉진하기 위해 공간군 분류 Kaggle 대회를 시작합니다. 참가자는 학습한 모델로 testNOtgt 데이터에 대한 예측을 제출하고 리더보드에 등록할 수 있습니다. 자세한 내용은 Kaggle 대회 페이지를 참조하십시오.

데이터 읽기:

```Python
from ase.db import connect

databs = connect("./binxrd.db")

for row in databs.select():
    atoms = row.toatoms()
    element = atoms.get_chemical_symbols()
    latt_dis = eval(getattr(row, 'latt_dis'))
    intensity = eval(getattr(row, 'intensity'))
    spg = eval(getattr(row, 'tager'))[0]
    crysystem = eval(getattr(row, 'tager'))[1]
```

## 튜토리얼

- **학습**: [model_tutorial](../tutorial/template.ipynb)
- **시뮬레이션**: [sim_tutorial](../sim/XRD.ipynb)
- **고처리량 시뮬레이션**: [HTsim_tutorial](../sim/tutorial_sim.ipynb)

## 결정 데이터

정리된 결정 데이터베이스가 필요하면 다음을 방문하십시오: https://huggingface.co/datasets/caobin/CrystDB

데이터셋 구성:

Database: [test_binxrd]  
Description: test_binxrd 데이터베이스에는 d-I 형식의 XRD 시뮬레이션 스펙트럼 119,569*2개가 포함되어 있습니다. 각 결정은 하나의 스펙트럼에 대응하는 테스트 데이터셋입니다.

Database: [train_binxrd]  
Description: train_binxrd 데이터베이스에는 d-I 형식의 XRD 시뮬레이션 스펙트럼 119,569*30개가 포함되어 있습니다. 각 결정은 각 파일에서 5개의 스펙트럼에 대응하는 학습 데이터셋입니다.

Database: [val_binxrd]  
Description: val_binxrd 데이터베이스에는 d-I 형식의 XRD 시뮬레이션 스펙트럼 119,569개가 포함되어 있습니다. 각 결정은 하나의 스펙트럼에 대응하는 검증 데이터셋입니다.

Database: [testNOtgt]  
Description: testNOtgt 데이터베이스에는 d-I 형식의 XRD 시뮬레이션 스펙트럼 119,569개가 포함되어 있습니다. 목표 변수가 없으며 순서는 무작위입니다.

Croissant로 리뷰 데이터 가져오기:

```Python
import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

dataset_info = mlc.Dataset(url).metadata.to_json()
print(dataset_info)
```

## 기여자

<a href="https://github.com/Bin-Cao/SimXRD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bin-Cao/SimXRD" />
</a>
