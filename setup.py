from setuptools import setup

setup(
   name='varvar',
   version='1.0',
   description='Model variance',
   author='Dror Speiser',
   packages=['varvar'],  # same as name
   install_requires=['scipy',
                     'numpy',
                     'numba',
                     ],  # external packages as dependencies
)
