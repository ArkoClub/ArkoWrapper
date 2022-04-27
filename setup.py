#!/usr/bin/env python
# coding: utf-8

from setuptools import setup

setup(
    name='arko-wrapper',
    version='0.0.1',
    author='Karako',
    author_email='karkaohear@gmail.com',
    url='',
    description=u'吃枣药丸',
    packages=['jujube_pill'],
    install_requires=[],
    entry_points={
        'console_scripts': [
            'jujube=jujube_pill:jujube',
            'pill=jujube_pill:pill'
        ]
    }
)