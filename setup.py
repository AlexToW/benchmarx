from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="benchmarx",
    version="0.0.11",
    description="Tools for benchmarking optimization methods",
    #package_dir={"" : "benchmarx"},
    #py_modules=['benchmarx'],
    packages=find_packages(),
    #packages=find_packages(where="benchmarx"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexToW/benchmarx",
    author="AlexToW",
    author_email="31salex31@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "hatchling",
        "jax",
        "jaxopt >= 0.6",
        "typing-extensions >= 4.5.0",
        "wandb >= 0.14.2",
        "pandas == 1.5.3",
        "plotly >= 5.14.1",
        "flax >= 0.6.10",
        "tensorflow_datasets"
    ],
    python_requires=">=3.7",
)
