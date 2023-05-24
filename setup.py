import os
import io
from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "HOOMD-blue-based package for coarse-grained polymer and chromosome simulations."


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != "" and not (req.startswith("#") or req.startswith("-"))
    ]

install_requires = get_requirements("requirements.txt")

setup(
    name="polychrom-hoomd",
    version=VERSION,
    description=DESCRIPTION,
    url="https://github.com/open2c/polychrom-hoomd",
    author="Open2C",
    author_email="open.chromosome.collective@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires
)
