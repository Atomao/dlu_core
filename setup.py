# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["dlu_core", "dlu_core.object_detection", "dlu_core.object_detection.yolo"]

package_data = {"": ["*"]}

install_requires = [
    "albumentations>=1.3.1,<2.0.0",
    "black>=23.11.0,<24.0.0",
    "hydra-core>=1.3.2,<2.0.0",
    "isort>=5.12.0,<6.0.0",
    "matplotlib>=3.8.2,<4.0.0",
    "numpy>=1.26.2,<2.0.0",
    "pandas>=2.1.3,<3.0.0",
    "pylint>=3.0.2,<4.0.0",
    "tqdm>=4.66.1,<5.0.0",
]

setup_kwargs = {
    "name": "dlu_core",
    "version": "0.1.1",
    "description": "My utility functions for work",
    "long_description": "",
    "author": "Danylo Kunyk",
    "author_email": "kunyk1507@gmail.com",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.9,<4.0",
}


setup(**setup_kwargs)
