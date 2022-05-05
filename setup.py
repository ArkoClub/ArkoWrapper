#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os

from setuptools import (
    find_packages,
    setup,
)

packages = find_packages('src')

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name='arko-wrapper',
    description="给你的Python迭代器加上魔法",
    version='0.0.4',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Arko',
    author_email='arko.space.cc@gmail.com',
    python_requires='>=3.9',
    url='https://github.com/ArkoClub/ArkoWrapper',
    packages=packages,
    package_dir={"": "src"},
    install_requires=[],
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ]
)
