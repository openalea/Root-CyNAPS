#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from os import walk
from os.path import abspath, normpath, splitext
from os.path import join as pj

from setuptools import setup, find_packages

short_descr = "Root-CyNAPS model"
readme = open('README.md').read()

# find packages
pkgs = find_packages('root_cynaps')

pkg_data = {}

nb = len(normpath(abspath("root_cynaps"))) + 1
data_rel_pth = lambda pth: normpath(abspath(pth))[nb:]

data_files = []
for root, dnames, fnames in walk("root_cynaps"):
    for name in fnames:
        if splitext(name)[-1] in [u'.json', u'.ini']:
            data_files.append(data_rel_pth(pj(root, name)))

pkg_data['root_cynaps'] = data_files

setup_kwds = dict(
    name='root_cynaps',
    version="0.0.1",
    description=short_descr,
    long_description=readme,
    author="Tristan Gérault",
    author_email="tristan.gerault@inrae.fr",
    url='',
    license='cecill-c',
    zip_safe=False,

    packages=pkgs,
    #namespace_packages=['openalea'],
    package_dir={'': 'root_cynaps'},

    package_data=pkg_data,
    setup_requires=[
        "pytest-runner",
    ],
    install_requires=[
    ],
    tests_require=[
        "pytest",
        "pytest-mock",
    ],
    entry_points={},
    keywords='',
)
# #}
# change setup_kwds below before the next pkglts tag

# do not change things below
# {# pkglts, pysetup.call
setup_kwds['setup_requires'] = []
setup_kwds['tests_requires'] = []

setup(**setup_kwds)
# #}
