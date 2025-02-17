#! /bin/bash

python setup.py sdist bdist_wheel

# 安装 pip install setuptools wheel twine
twine upload dist/*

# 输入pypi token即可