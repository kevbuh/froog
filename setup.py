
#!/usr/bin/env python3
# this file specifies how the froog package is installed, including any necessary dependencies required to run

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='froog',
      version='0.2.8',
      description='a beautifully simplistic ml framework',
      author='Kevin Buhler',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['froog'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy', 'requests', 'matplotlib'],
      python_requires='>=3.8',
      include_package_data=True)