from setuptools import setup, find_packages

setup(
    name="twitter-topic-classifier",
    version="1.0.0",
    author="Ibrahim Akintunde Akinyera",
    author_email="akinyeraakintunde@gmail.com",
    description="A machine learning pipeline for multi-class social media text classification (Politics, Sports, Entertainment).",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("docs", "data",)),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "nltk",
        "emoji"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
)