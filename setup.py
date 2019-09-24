from os import path
from setuptools import setup, find_packages

setup(
    name='easysurrogate',

    version=0.1,

    description=(''),

#    url='https://readthedocs.org/projects/easyvvuq/',

#    author='CCS',

    install_requires=['numpy'],

    packages=find_packages(),

    include_package_data=True,
)