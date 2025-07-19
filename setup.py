#!/usr/bin/env python3
"""
Setup script for Crypto Trading Analysis Package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto-trading-analysis",
    version="1.0.0",
    author="Crypto Analyst",
    author_email="analyst@crypto-project.com",
    description="Comprehensive cryptocurrency trading analysis with pattern recognition and machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-trading-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-analysis=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="cryptocurrency, trading, analysis, machine learning, technical analysis, pattern recognition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/crypto-trading-analysis/issues",
        "Source": "https://github.com/yourusername/crypto-trading-analysis",
        "Documentation": "https://github.com/yourusername/crypto-trading-analysis/blob/main/README.md",
    },
) 