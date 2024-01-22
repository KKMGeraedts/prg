from setuptools import setup, find_packages

setup(
        name="prg",
        version="0.1.1",
        packages=find_packages(),
        description="Implementation of the Phenomenological Renormalization Group procedure.",
        long_description=open("README.md").read(),
        url="https://github.com/KKMGeraedts/prg",
        author="Karel Geraedts",
        install_requires=[]
)
