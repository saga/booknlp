from setuptools import find_packages, setup

setup(
    name="booknlp",
    version="1.0.8",
    packages=find_packages(),
    python_requires=">=3.12,<3.13",
    url="https://github.com/saga/booknlp",
    author="David Bamman",
    author_email="dbamman@berkeley.edu",
    include_package_data=True,
    license="MIT",
    install_requires=[
        "httpx[socks]>=0.28.1",
        "numpy>=2.0,<3",
        "spacy>=3.7,<4",
        "torch>=2.2,<3",
        "transformers>=4.40,<5",
    ],
    extras_require={
        "coref": ["maverick-coref==1.0.7"],
        "ner": ["gliner>=0.2,<1"],
    },
)
