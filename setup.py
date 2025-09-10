#!/usr/bin/env python

import os
import platform

from setuptools import find_packages, setup

# Run with python setup.py develop --user

setup_requires = []

scripts = []
# scripts.extend(glob('scripts/*py'))


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering
"""
setup(
    name="nn_factor",
    version="0.12",
    packages=find_packages(),  # <= will grab nn_factor + subfolders
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "numpy>=2.2.6",
        "pandas>=2.3.2",
        "tensorflow>=2.20.0",
        "sklearn>=1.7.2"
    ],
    scripts=scripts,
    author="Arthur Sugden",
    author_email="sugdena@duq.edu",
    description="Pure math models for partitions",
    setup_requires=setup_requires,
    # setup_requires=['setuptools_cython'],
    url="https://allref.io",
    platforms=["Linux", "Mac OS-X", "Windows"],
    ext_modules=[],
)
