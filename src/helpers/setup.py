"""Building mean IoU module"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("./helpers/*.pyx"),
    include_dirs=[numpy.get_include()],
    package_dir="./helpers/",
)
