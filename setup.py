from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()
    

setup(
    name='varvar',
    version='1.0.0',
    description='Model variance with multiplicative variance trees',
    url="https://github.com/drorspei/varvar",
    author='Dror Speiser',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['varvar'],
    install_requires=[
        'numpy',
        'numba',
    ],
)
