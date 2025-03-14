#! /bin/bash

# 获取脚本所在目录并切换到该目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR" || exit

python setup.py sdist bdist_wheel

# 安装 pip install setuptools wheel twine
twine upload dist/*

# 输入pypi token即可
