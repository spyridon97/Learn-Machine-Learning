import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="Learn-Machine-Learning",
    version="0.0.1",
    author="Spyridon Tsalikis",
    author_email="spyridon97@hotmail.com",
    description="A collection of algorithms to learn Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spyridon97/Learn-Machine-Learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
