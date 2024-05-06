from setuptools import setup, find_packages

setup(
        name="prg",
        version="0.1.3",
        packages=find_packages(),
        description="Implementation of the Phenomenological Renormalization Group procedure.",
        long_description=open("README.md").read(),
        url="https://github.com/KKMGeraedts/prg",
        author="Karel Geraedts",
        install_requires=[
            "numpy>=1.26.3",
            "matplotlib>=3.8.2",
            "scipy>=1.12.0",
            "pandas>=2.1.4"
            ]
)
