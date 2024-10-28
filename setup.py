from setuptools import find_packages, setup

setup(
    name="Distributed Machine Learning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for safe serialization of genetic programming primitives",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mekaneeky/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "deap>=1.3.1",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.3",
            "black>=21.5b1",
            "isort>=5.8.0",
            "flake8>=3.9.1",
        ],
    },
)
