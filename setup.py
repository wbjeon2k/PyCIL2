from setuptools import setup, find_packages

setup(
    name="pycil2",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch >= 1.8.1",
        "torchvision >= 0.6.0",
        "tqdm",
        "numpy",
        "scipy",
        "quadprog",
        "POT",
    ],
    python_requires=">=3.8",
)