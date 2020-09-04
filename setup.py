import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="centpy",
    version="0.1",
    author="Anil Zenginoglu",
    author_email="anil@umd.edu",
    description="A numerical solver for conservation laws based on central schemes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnilZen/centpy",
    packages=setuptools.find_packages(exclude=["*.tests"]),
    #    ("centpy",),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
