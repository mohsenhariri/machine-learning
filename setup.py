from setuptools import setup, find_packages, find_namespace_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ml",
    version="1.0",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mohsen Hariri",
    author_email="m.hariri@sharif.edu",
    # packages=find_packages(),
    # packages=find_namespace_packages() # for all
    packages=find_namespace_packages(include=["architectures.*"])
    # install_requires=[],
    # python_requires="~=3.3",
)
