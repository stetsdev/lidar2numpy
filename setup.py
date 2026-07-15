"""Setuptools extension definition for the optional Cython decoder backend."""

from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "lidar2numpy._decoder_core",
        ["src/lidar2numpy/_decoder_core.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
