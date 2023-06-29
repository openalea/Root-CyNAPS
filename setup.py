#!/usr/bin/env python
# -*- coding: utf-8 -*-


from os import walk
from os.path import abspath, normpath, splitext
from os.path import join as pj

from setuptools import setup, find_packages

short_descr = "Root Cycling Nitrogen Across Plant Scales : a FSPM root nitrogen cycle model"
readme = open('README.md').read()
#history = open('HISTORY.rst').read()

# find packages
pkgs = find_packages('.')

pkg_data = {}

nb = len(normpath(abspath("rhydromin"))) + 1
data_rel_pth = lambda pth: normpath(abspath(pth))[nb:]

data_files = []
for root, dnames, fnames in walk("rhydromin"):
    for name in fnames:
        if splitext(name)[-1] in [u'.json', u'.ini']:
            data_files.append(data_rel_pth(pj(root, name)))

pkg_data['rhydromin'] = data_files

setup_kwds = dict(
    name='Root-CyNAPS',
    version="0.0.1",
    description=short_descr,
    long_description=readme + '\n\n', # + history,
    author="Tristan Gérault",
    author_email="tristan.gerault@inrae.fr",
    url='',
    license='cecill-c',
    zip_safe=False,

    packages=pkgs,
    #namespace_packages=['openalea'],
    #package_dir={'': '.'},

    package_data=pkg_data,
    install_requires=[],

    entry_points={},
    keywords='',
)

setup(**setup_kwds)

