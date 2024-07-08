from os import path
from setuptools import setup, find_packages

setup(
    name='easysurrogate',

<<<<<<< HEAD
    version='0.25',
=======
    version='0.24.3',
>>>>>>> master

    description=(''),

#    url='https://readthedocs.org/projects/easyvvuq/',

    author='CWI',

    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'scikit-learn', 'mogp_emulator', 'tqdm'],
    
    #package_dir={'':'easysurrogate'},

    packages=find_packages(),

    include_package_data=True,
)
