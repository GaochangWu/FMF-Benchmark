from setuptools import setup, find_packages

setup(
    name='FMF',
    packages=find_packages(exclude=("configs", "tests")),
)
