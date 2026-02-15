from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="robust-mixed-dist",
    version="0.1.18",
    author="Fabio Scielzo Ortiz",
    author_email="fabio.scielzoortiz@gmail.com",
    description="Compute statistical robust distances for mixed data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FabioScielzoOrtiz/robust_mixed_dist-package",  # add your project URL here
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['polars', 'numpy', 'pandas', 'scipy<=1.16.2'],
    python_requires=">=3.7"
)
