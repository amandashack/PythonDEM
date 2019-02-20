import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SparseInvDEM",
    version="0.0.1",
    author="Amanda Shackelford",
    author_email="ajshacke@uci.edu",
    description="A program to solve for the DEM of solar plasma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amandashack/PythonDEM",
    packages=setuptools.find_packages(),
    install_requires = ['numpy', 'matplotlib', 'requests'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)