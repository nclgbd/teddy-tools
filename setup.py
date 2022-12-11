"""
`setup.py` for `teddytools`
"""
import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="teddytools",
    version="0.0.1.dev0",
    packages=setuptools.find_packages(),
    description="Teddy tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
