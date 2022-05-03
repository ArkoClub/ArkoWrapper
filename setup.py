#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os

from setuptools import (
    find_packages,
    setup,
)

NAME = 'arko-wrapper'
DESCRIPTION = 'Make your Python iterators more magical.'
URL = 'https://github.com/ArkoClub/ArkoWrapper'
EMAIL = 'arko.space.cc@gmail.com'
AUTHOR = 'Arko'
REQUIRES_PYTHON = '>=3.9'
VERSION = '0.0.2'

REQUIRED = []
EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
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
