# SimXRD-4M [ICLR 2025](https://iclr.cc/virtual/2025/poster/28452)

<p align="center">
  <strong>Idioma:</strong>
  <a href="../README.md">English</a> | <a href="README.zh-CN.md">中文</a> | <a href="README.ja.md">日本語</a> | <a href="README.ko.md">한국어</a> | <a href="README.de.md">Deutsch</a> | Español
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
> **SimXRD es el primer conjunto de datos fundacional realmente a gran escala para la investigación de difracción de rayos X impulsada por IA.**  
> Antes de SimXRD, el progreso del aprendizaje automático en cristalografía estaba limitado por la falta de conjuntos de datos de difracción suficientemente grandes, físicamente realistas y sistemáticamente evaluados. SimXRD cambió este panorama al introducir más de 4 millones de patrones XRD simulados de alta fidelidad, que abarcan más de 119.000 estructuras cristalinas bajo 33 condiciones experimentales físicamente diversas. A diferencia de las bases de datos de difracción convencionales, que contienen patrones limitados o idealizados, SimXRD modela explícitamente variaciones del mundo real, como el ensanchamiento de picos, la perturbación de la red, los efectos instrumentales y las transformaciones que preservan la simetría. Además de ser un conjunto de datos, SimXRD estableció uno de los primeros benchmarks estandarizados a gran escala para el aprendizaje de representaciones de difracción, permitiendo entrenar, evaluar y escalar modelos modernos de IA para el análisis cristalográfico. Este trabajo proporciona una base de datos clave para sistemas emergentes como [XQueryer](https://github.com/Bin-Cao/XQueryer) y [XDecomposer](https://github.com/Licht0812/XDecomposer).

**Descripción de los datos:** Los cristales se clasifican en 230 grupos espaciales, cada uno de los cuales representa una categoría de simetría distinta. Los patrones XRD, correspondientes a la estructura cristalina, son herramientas esenciales para estudiar materiales. Sin embargo, los patrones XRD se ven afectados por el entorno de medición, la instrumentación, la fuente de rayos X y las características de la muestra, como el tamaño de grano y la orientación. Por ello, presentan variaciones en intensidad, ensanchamiento de picos y otras características, lo que dificulta la identificación precisa de fases. Esta base de datos facilita el entrenamiento de modelos al proporcionar espectros de difracción bajo diversas condiciones ambientales. El objetivo final es que el modelo identifique con precisión el grupo espacial correcto a partir de patrones XRD.

---

## Instalación

Para procesar el archivo de base de datos se necesitan las siguientes bibliotecas:

- ase
- tqdm

```bash
pip install ase tqdm
```

Anuncio de competición en Kaggle

Para evaluar modelos avanzados y fomentar su desarrollo, lanzamos una competición de Kaggle para la clasificación de grupos espaciales. Los participantes pueden enviar sus predicciones sobre los datos testNOtgt usando sus modelos entrenados y aparecer en la clasificación. Para más detalles, consulte la página de la competición en Kaggle.

Lectura de datos:

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

## Tutoriales

- **entrenamiento**: [model_tutorial](../tutorial/template.ipynb)
- **simulación**: [sim_tutorial](../sim/XRD.ipynb)
- **simulación de alto rendimiento**: [HTsim_tutorial](../sim/tutorial_sim.ipynb)

## Datos cristalinos

Si necesita la base de datos cristalina organizada, visite: https://huggingface.co/datasets/caobin/CrystDB

Distribución del conjunto de datos:

Database: [test_binxrd]  
Description: La base test_binxrd contiene 119,569*2 espectros XRD simulados en formato d-I. Sirve como conjunto de prueba, donde cada cristal corresponde a un solo espectro.

Database: [train_binxrd]  
Description: La base train_binxrd contiene 119,569*30 espectros XRD simulados en formato d-I. Sirve como conjunto de entrenamiento, donde cada cristal corresponde a 5 espectros en cada archivo.

Database: [val_binxrd]  
Description: La base val_binxrd contiene 119,569 espectros XRD simulados en formato d-I. Sirve como conjunto de validación, donde cada cristal corresponde a un solo espectro.

Database: [testNOtgt]  
Description: La base testNOtgt contiene 119,569 espectros XRD simulados en formato d-I. Es un conjunto de prueba sin variable objetivo y con orden aleatorio.

Obtener datos de revisión con Croissant:

```Python
import mlcroissant as mlc
url = "https://huggingface.co/datasets/caobin/SimXRDreview/raw/main/simxrd_croissant.json"

dataset_info = mlc.Dataset(url).metadata.to_json()
print(dataset_info)
```

## Colaboradores

<a href="https://github.com/Bin-Cao/SimXRD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bin-Cao/SimXRD" />
</a>
