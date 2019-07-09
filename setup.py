import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sfpd",
    version="0.0.1",
    author="Andrew D. Robertson",
    author_email="andy.d.robertson@gmail.com",
    description="Tools for the detection of surprisingly frequent phrases in a corpus of text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andehr/sfpd",
    packages=setuptools.find_packages(),
    license="proprietary and confidential",
    install_requires=[
        "sklearn>=0.19.1",
        "spacy>=2.0.11"
        "numpy>=1.14.3"
        "pandas>=0.23.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
