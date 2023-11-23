# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['dlu_core']

package_data = \
{'': ['*']}

setup_kwargs = {
    'name': 'dlu_core',
    'version': '0.1.1',
    'description': 'My utility functions for work',
    'long_description': '',
    'author': 'Danylo Kunyk',
    'author_email': 'kunyk1507@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)

