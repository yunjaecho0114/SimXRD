# SimXRD-4M [ICLR 2025](https://iclr.cc/virtual/2025/poster/28452)

<p align="center">
  <strong>言語:</strong>
  <a href="../README.md">English</a> | <a href="README.zh-CN.md">中文</a> | 日本語 | <a href="README.ko.md">한국어</a> | <a href="README.de.md">Deutsch</a> | <a href="README.es.md">Español</a>
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
> **SimXRD は、AI 駆動型 X 線回折研究のための初の真に大規模な基盤データセットです。**  
> SimXRD 以前は、結晶学における機械学習の進展は、十分に大きく、物理的に現実的で、体系的にベンチマークされた回折データセットの不足によって制約されていました。SimXRD は、119,000 を超える結晶構造と 33 種類の物理的に多様な実験条件を対象に、400 万件を超える高忠実度の模擬 XRD パターンを提供することで、この状況を大きく変えました。従来の回折データベースが限られた、または理想化されたパターンを含むのに対し、SimXRD はピーク広がり、格子摂動、装置効果、対称性を保つ変換など、現実世界の変動を明示的にモデル化します。データセットにとどまらず、SimXRD は回折表現学習のための最初期の標準化された大規模ベンチマークの一つを確立し、結晶学解析に向けた現代的 AI モデルの訓練、評価、スケーリングを可能にしました。この成果は、[XQueryer](https://github.com/Bin-Cao/XQueryer) や [XDecomposer](https://github.com/Licht0812/XDecomposer) などの新しいシステムを支えるデータ基盤にもなっています。

**データ説明：** 結晶は 230 の空間群に分類され、それぞれが異なる対称性カテゴリを表します。結晶構造に対応する XRD パターンは、材料研究における重要な手段です。ただし、XRD パターンは測定環境、装置、X 線源、粒径や配向などの試料特性に影響されます。そのため、強度変化やピーク広がりなどの特徴が変化し、正確な相同定を難しくします。本データベースは、多様な環境条件下の回折スペクトルを提供することでモデル訓練を支援し、最終的には XRD パターンから正しい空間群を高精度に識別することを目指します。

---

## インストール

データベースファイルを処理するには、以下のライブラリが必要です。

- ase
- tqdm

```bash
pip install ase tqdm
```

Kaggle コンペティションのお知らせ

高度なモデルをベンチマークし、さらなる開発を促進するため、空間群分類の Kaggle コンペティションを開始します。参加者は、訓練済みモデルを用いて testNOtgt データに対する予測を提出し、リーダーボードに登録できます。詳細は Kaggle コンペティションページを参照してください。

データの読み込み：

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

## チュートリアル

- **学習**：[model_tutorial](../tutorial/template.ipynb)
- **シミュレーション**：[sim_tutorial](../sim/XRD.ipynb)
- **高スループットシミュレーション**：[HTsim_tutorial](../sim/tutorial_sim.ipynb)

## 結晶データ

整理済みの結晶データベースが必要な場合はこちらをご覧ください：https://huggingface.co/datasets/caobin/CrystDB

データセット構成：

Database: [test_binxrd]  
Description: test_binxrd には d-I 形式の XRD 模擬スペクトルが 119,569*2 件含まれます。各結晶は 1 つのスペクトルに対応するテストデータセットです。

Database: [train_binxrd]  
Description: train_binxrd には d-I 形式の XRD 模擬スペクトルが 119,569*30 件含まれます。各結晶は各ファイル内で 5 つのスペクトルに対応する訓練データセットです。

Database: [val_binxrd]  
Description: val_binxrd には d-I 形式の XRD 模擬スペクトルが 119,569 件含まれます。各結晶は 1 つのスペクトルに対応する検証データセットです。

Database: [testNOtgt]  
Description: testNOtgt には d-I 形式の XRD 模擬スペクトルが 119,569 件含まれます。ターゲット変数を含まず、順序はランダム化されています。

Croissant によるレビューデータの取得：

```Python
import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

dataset_info = mlc.Dataset(url).metadata.to_json()
print(dataset_info)
```

## コントリビューター

<a href="https://github.com/Bin-Cao/SimXRD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bin-Cao/SimXRD" />
</a>
