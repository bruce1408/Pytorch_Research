from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
    
setup(
    # readme file
    long_description=long_description,
    long_description_content_type='text/markdown',
    # package information
    
    name="spectrautils",
    version="0.3.3",
    packages=find_packages(),
    description="A powerful tools for python",
    author="bruce_cui",
    author_email="summer56567@163.com",
    install_requires=[
        # 依赖列表
        "termcolor >= 2.3.0",
        "colorama >= 0.4.4",
        "onnxruntime-gpu >= 1.18.0",
        "onnx >= 1.16.1",
        "tqdm >= 4.67.1",
        "pandas >= 1.5.3",
        "holoviews >= 1.17.1",
        "hvplot >= 0.10.0"
    ],
)
