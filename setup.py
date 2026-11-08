# Copyright 2025 w.Sanice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from setuptools import setup, find_packages

with open("sanice/__init__.py", "r", encoding="utf-8") as fh:
    version = re.search(r'__version__ = "([^"]+)"', fh.read()).group(1)

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Biblioteca fluente para Data Science ágil (ETL, AutoML e gráficos)."

setup(
    name="sanice",
    version=version,
    author="wSanice",
    author_email="wansanice@proton.me",
    description="Biblioteca fluente para Data Science ágil (ETL, AutoML e gráficos).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wSanice/sanice",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "joblib>=1.2.0",
        "openpyxl>=3.1.0",
        "pyarrow>=11.0.0",
        "sqlalchemy>=2.0.0",
        "unidecode>=1.3.0"
    ],

    extras_require={
        "api": ["fastapi>=0.95.0", "uvicorn>=0.22.0", "pydantic>=1.10.0"],
        "db": ["pymongo", "psycopg2-binary","pymysql"],
        "dev": ["pytest", "twine", "wheel", "pytest-mock", "coverage", "httpx"]
    },
    entry_points={
        "console_scripts": [
            "sanice=sanice.core:cli",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/wSanice/sanice",
        "Source": "https://github.com/wSanice/sanice",
        "Funding": "https://github.com/sponsors/wSanice",
    },
)
