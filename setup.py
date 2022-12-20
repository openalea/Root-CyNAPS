#!/usr/bin/env python
# -*- coding: utf-8 -*-


from os import walk
from os.path import abspath, normpath, splitext
from os.path import join as pj

from setuptools import setup, find_packages

short_descr = "RhizoDeposition"
readme = open('README.md').read()
#history = open('HISTORY.rst').read()

# find packages
pkgs = find_packages('.')

pkg_data = {}

nb = len(normpath(abspath("rhizodep"))) + 1
data_rel_pth = lambda pth: normpath(abspath(pth))[nb:]

data_files = []
for root, dnames, fnames in walk("rhizodep"):
    for name in fnames:
        if splitext(name)[-1] in [u'.json', u'.ini']:
            data_files.append(data_rel_pth(pj(root, name)))

pkg_data['rhizodep'] = data_files

setup_kwds = dict(
    name='rhizodep',
    version="0.0.2",
    description=short_descr,
    long_description=readme + '\n\n', # + history,
    author="Frederic Rees",
    author_email="frederic.rees@inrae.fr",
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

