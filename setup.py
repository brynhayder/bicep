import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicep",
    version="1.0.1",
    author="brynhayder",
    description="Tools for training PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brynhayder/bicep",
    packages=setuptools.find_packages(),
    # The following may be untrue and needs to be checked!
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
