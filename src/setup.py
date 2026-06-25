"""Package configuration for Pysimxrd."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


setup(
    name="Pysimxrd",
    version="1.0.0",
    description="Physical simulation of powder X-ray diffraction patterns",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Cao Bin",
    author_email="bcao686@connect.hkust-gz.edu.cn",
    url="https://github.com/Bin-Cao/SimXRD",
    project_urls={
        "Homepage": "https://bin-cao.github.io",
        "Source": "https://github.com/Bin-Cao/SimXRD",
    },
    license="MIT",
    packages=find_packages(),
    package_data={
        "Pysimxrd": ["CGCNN_atom_emb.json"],
    },
    include_package_data=True,
    install_requires=[
        "ase",
        "numpy",
        "pandas",
        "pymatgen",
        "scipy",
        "spglib",
        "sympy",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "xrd",
        "powder diffraction",
        "crystallography",
        "materials science",
        "simulation",
    ],
)
