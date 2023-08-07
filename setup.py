
#!/usr/bin/env python3
# this file specifies how the ribbit package is installed, including any necessary dependencies required to run

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='ribbit',
      version='0.2.3',
      description='INSANELY SIMPLE AI/ML FRAMEWORK',
      author='Kevin Buhler',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['ribbit'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy', 'requests'],
      python_requires='>=3.8',
      include_package_data=True)