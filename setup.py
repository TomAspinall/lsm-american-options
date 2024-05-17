
import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="lsm-american-options",
    version="0.0.0",
    author="Tom Aspinall",
    author_email="tomaspinall2512@gmail.com",
    description="Financial and real option pricing through Least-Squares Monte Carlo Simulation",
    packages=find_packages(),
    python_requires=">=3.6",
    # install_requires=,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
