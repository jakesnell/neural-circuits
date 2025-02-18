# -*- coding: utf-8 -*-

# Adapted from https://github.com/navdeep-G/samplemod/
# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

setup(
    name="partikel",
    version="0.1.0",
    description="Particle filtering using PyTorch",
    long_description=readme,
    author="",
    author_email="",
    url="",
    license="",
    packages=find_packages(exclude=("tests")),
)
