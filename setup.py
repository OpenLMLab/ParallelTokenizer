# Copyright (c) OpenLMLab. All rights reserved.
import os

from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)


def readme():
    with open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    with open(os.path.join(pwd, "version.txt"), "r") as f:
        content = f.read()
    return content


setup(
    name="ParallelTokenizer",
    version=get_version(),
    description="a tool that uses a parallel approach to tokenizer to achieve superior acceleration",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
)
