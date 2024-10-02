import os
import io

from setuptools import setup, find_packages


VERSION = "0.1.0"
DESCRIPTION = "HOOMD-blue-based package for coarse-grained polymer and chromosome simulations."


setup(
    name="polychrom-hoomd",
    version=VERSION,
    description=DESCRIPTION,
    url="https://github.com/open2c/polychrom-hoomd",
    author="Open2C",
    author_email="open.chromosome.collective@gmail.com",
    license="MIT",
    packages=find_packages()
)
