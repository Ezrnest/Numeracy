# Created by lyc at 2020/10/18 18:57
import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numeracy",  # Replace with your own username
    version="0.0.2",
    author="Example Author",
    author_email="1403718476@qq.com",
    description="A numerical algorithm package in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/140378476/Numeracy",
    install_requires=[
        'numpy>=1.17'
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
