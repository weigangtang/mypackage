import setuptools

setuptools.setup(
    name="mypackage",
    version="0.0.1",
    author="victor tang",
    author_email="tangw5@mcmaster.ca",
    description="first try to develop python package with github",
    long_description='no long description',
    url="https://github.com/weigangtang/mypacakge",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)