# SimXRD-4M [ICLR 2025](https://iclr.cc/virtual/2025/poster/28452)

<p align="center">
  <strong>语言:</strong>
  <a href="../README.md">English</a> | 中文 | <a href="README.ja.md">日本語</a> | <a href="README.ko.md">한국어</a> | <a href="README.de.md">Deutsch</a> | <a href="README.es.md">Español</a>
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
> **SimXRD 是首个真正大规模、面向 AI 驱动 X 射线衍射研究的基础数据集。**  
> 在 SimXRD 之前，晶体学机器学习的发展长期受限于缺少足够大、物理真实且经过系统基准化的衍射数据集。SimXRD 通过提供超过 400 万条高保真模拟 XRD 图谱，覆盖 119,000 多个晶体结构和 33 种具有物理差异的实验条件，显著改变了这一局面。不同于传统衍射数据库中有限或理想化的图谱，SimXRD 明确建模了现实世界中的峰展宽、晶格扰动、仪器效应和保持对称性的结构变换。除数据集之外，SimXRD 还建立了首批用于衍射表征学习的大规模标准化基准之一，使现代 AI 模型能够在晶体学分析任务中进行训练、评估和扩展。这项工作为下一代 AI 驱动的衍射理解奠定了关键基础设施，也成为 [XQueryer](https://github.com/Bin-Cao/XQueryer) 和 [XDecomposer](https://github.com/Licht0812/XDecomposer) 等系统背后的数据基础。

**数据说明：** 晶体被划分为 230 个空间群，每个空间群代表一种不同的对称类别。XRD 图谱与晶体结构对应，是研究材料的重要工具。然而，XRD 图谱会受到测试环境、仪器、X 射线光源以及样品特征，如晶粒尺寸和取向等因素影响。因此，图谱会出现强度变化、峰展宽等差异，给准确相鉴定带来挑战。本数据库旨在通过提供多种环境条件下的衍射谱数据来支持模型训练，最终目标是让模型能够根据 XRD 图谱准确识别正确的空间群。

---

## 安装

处理数据库文件需要安装以下库：

- ase
- tqdm

```bash
pip install ase tqdm
```

Kaggle 竞赛公告

为了对先进模型进行基准测试并推动其发展，我们将启动一个空间群分类 Kaggle 竞赛。参赛者可以使用训练好的模型，对 testNOtgt 数据提交预测结果并进入排行榜。更多详细信息请访问 Kaggle 竞赛页面。

读取数据：

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

    # element, a list, e.g., ['C', 'H', 'O']
    # latt_dis, a list, lattice plane distances
    # intensity, a list, diffraction intensity
    # spg, int, space group number
    # crysystem, int, crystal system number
```

## 教程

- **训练**：[model_tutorial](../tutorial/template.ipynb)
- **模拟**：[sim_tutorial](../sim/XRD.ipynb)
- **高通量模拟**：[HTsim_tutorial](../sim/tutorial_sim.ipynb)

## 晶体数据

如果需要整理好的晶体数据库，请访问：https://huggingface.co/datasets/caobin/CrystDB

数据集分布：

Database: [test_binxrd]  
Description: test_binxrd 数据库包含 119,569*2 条 d-I 格式的 XRD 模拟谱。该数据集用于测试，每个晶体只对应一条谱。

Database: [train_binxrd]  
Description: train_binxrd 数据库包含 119,569*30 条 d-I 格式的 XRD 模拟谱。该数据集用于训练，每个晶体在每个文件中对应 5 条谱。

Database: [val_binxrd]  
Description: val_binxrd 数据库包含 119,569 条 d-I 格式的 XRD 模拟谱。该数据集用于验证，每个晶体只对应一条谱。

Database: [testNOtgt]  
Description: testNOtgt 数据库包含 119,569 条 d-I 格式的 XRD 模拟谱。该测试数据集没有目标变量，并且顺序已随机打乱。

使用 Croissant 获取评审数据：

```Python
import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

dataset_info = mlc.Dataset(url).metadata.to_json()
print(dataset_info)

from dataset.parse import load_dataset, bar_progress
for file_info in dataset_info['distribution']:
    wget.download(file_info['contentUrl'], './', bar=bar_progress)

from torch.utils.data import DataLoader

train_loader = DataLoader(load_dataset(name='train.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(load_dataset(name='val.tfrecord'), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
test_loader = DataLoader(load_dataset(name='test.tfrecord'), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
```

## 贡献者

<a href="https://github.com/Bin-Cao/SimXRD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bin-Cao/SimXRD" />
</a>
