# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="multicellmimo",
    version="0.1",
    description="Multi-cell MIMO optimization tools",
    long_description=readme,
    author="Andreas Dotzler",
    author_email="contact@andreasdotzler.info",
    url="https://github.com/andreasdotzler/TODO",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    install_require=['cvxpy', 'numpy']
)