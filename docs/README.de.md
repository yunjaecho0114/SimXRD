# SimXRD-4M [ICLR 2025](https://iclr.cc/virtual/2025/poster/28452)

<p align="center">
  <strong>Sprache:</strong>
  <a href="../README.md">English</a> | <a href="README.zh-CN.md">中文</a> | <a href="README.ja.md">日本語</a> | <a href="README.ko.md">한국어</a> | Deutsch | <a href="README.es.md">Español</a>
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
> **SimXRD ist der erste wirklich großskalige Foundation-Datensatz für KI-gestützte Röntgenbeugungsforschung.**  
> Vor SimXRD wurde der Fortschritt des maschinellen Lernens in der Kristallographie grundlegend durch das Fehlen ausreichend großer, physikalisch realistischer und systematisch benchmarkbarer Beugungsdatensätze begrenzt. SimXRD verändert diese Situation durch mehr als 4 Millionen hochgetreue simulierte XRD-Muster aus über 119.000 Kristallstrukturen unter 33 physikalisch unterschiedlichen experimentellen Bedingungen. Im Gegensatz zu klassischen Beugungsdatenbanken mit begrenzten oder idealisierten Mustern modelliert SimXRD reale Variationen wie Peakverbreiterung, Gitterstörungen, instrumentelle Effekte und symmetrieerhaltende Transformationen explizit. Darüber hinaus etablierte SimXRD einen der ersten standardisierten großskaligen Benchmarks für Repräsentationslernen in der Beugung und ermöglicht Training, Evaluation und Skalierung moderner KI-Modelle für kristallographische Analysen. Diese Arbeit bildet eine wichtige Datengrundlage für Systeme wie [XQueryer](https://github.com/Bin-Cao/XQueryer) und [XDecomposer](https://github.com/Licht0812/XDecomposer).

**Datenbeschreibung:** Kristalle werden in 230 Raumgruppen eingeteilt, von denen jede eine eigene Symmetriekategorie darstellt. XRD-Muster entsprechen der Kristallstruktur und sind wichtige Werkzeuge zur Untersuchung von Materialien. Sie werden jedoch durch Messumgebung, Instrumentierung, Röntgenquelle und Probeneigenschaften wie Korngröße und Orientierung beeinflusst. Dadurch entstehen Unterschiede in Intensität, Peakverbreiterung und weiteren Merkmalen, was eine genaue Phasenidentifikation erschwert. Diese Datenbank unterstützt das Modelltraining durch Beugungsspektren unter unterschiedlichen Umgebungsbedingungen. Ziel ist es, dass Modelle anhand von XRD-Mustern die korrekte Raumgruppe zuverlässig identifizieren.

---

## Installation

Für die Verarbeitung der Datenbankdatei werden folgende Bibliotheken benötigt:

- ase
- tqdm

```bash
pip install ase tqdm
```

Kaggle-Wettbewerb

Um fortgeschrittene Modelle zu benchmarken und ihre Weiterentwicklung zu fördern, starten wir einen Kaggle-Wettbewerb zur Raumgruppenklassifikation. Teilnehmende können Vorhersagen für die testNOtgt-Daten mit ihren trainierten Modellen einreichen und auf dem Leaderboard vergleichen. Weitere Informationen finden Sie auf der Kaggle-Wettbewerbsseite.

Daten lesen:

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

## Tutorials

- **Training**: [model_tutorial](../tutorial/template.ipynb)
- **Simulation**: [sim_tutorial](../sim/XRD.ipynb)
- **Hochdurchsatzsimulation**: [HTsim_tutorial](../sim/tutorial_sim.ipynb)

## Kristalldaten

Die organisierte Kristalldatenbank finden Sie hier: https://huggingface.co/datasets/caobin/CrystDB

Datensatzverteilung:

Database: [test_binxrd]  
Description: Die test_binxrd-Datenbank enthält 119,569*2 simulierte XRD-Spektren im d-I-Format. Sie dient als Testdatensatz, wobei jeder Kristall genau einem Spektrum entspricht.

Database: [train_binxrd]  
Description: Die train_binxrd-Datenbank enthält 119,569*30 simulierte XRD-Spektren im d-I-Format. Sie dient als Trainingsdatensatz, wobei jeder Kristall pro Datei fünf Spektren besitzt.

Database: [val_binxrd]  
Description: Die val_binxrd-Datenbank enthält 119,569 simulierte XRD-Spektren im d-I-Format. Sie dient als Validierungsdatensatz, wobei jeder Kristall genau einem Spektrum entspricht.

Database: [testNOtgt]  
Description: Die testNOtgt-Datenbank enthält 119,569 simulierte XRD-Spektren im d-I-Format. Dieser Testdatensatz enthält keine Zielvariable und ist zufällig sortiert.

Review-Daten mit Croissant abrufen:

```Python
import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

dataset_info = mlc.Dataset(url).metadata.to_json()
print(dataset_info)
```

## Mitwirkende

<a href="https://github.com/Bin-Cao/SimXRD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bin-Cao/SimXRD" />
</a>
