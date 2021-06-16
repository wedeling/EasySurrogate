from os import path
from setuptools import setup, find_packages

setup(
    name='easysurrogate',

    version=0.16,

    description=(''),

#    url='https://readthedocs.org/projects/easyvvuq/',

    author='CWI',

    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'sklearn', 'mogp_emulator'],
    
    #package_dir={'':'easysurrogate'},

    packages=find_packages(),

    include_package_data=True,
)
