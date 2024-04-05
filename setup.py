
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="lsm-american-options",
    version="0.0.1",
    author="Tom Aspinall",
    author_email="tomaspinall2512@gmail.com",
    description="American Option Pricing through Least-Squares Monte Carlo Simulation",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=["numpy"],
    long_description=long_description,
    long_description_content_type="text/markdown",
)