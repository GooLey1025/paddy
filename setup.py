from setuptools import setup, find_packages

setup(
    name="paddy",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow",
        "numpy",
        "pyyaml",
    ],
    python_requires=">=3.6",
)
