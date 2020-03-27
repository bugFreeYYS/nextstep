import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nextstep",
    version="0.0.2",
    author="Yang Yuesong",
    author_email="yangyuesongyys@gmail.com",
    description="USEP price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YangYuesong0323/EMC",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires = [
        'dataflows == 0.071',
        'wwoh-hist == 0.0.4',
        'tenserflow == 1.1.0',
        'statsmodels == 0.10.1'
        ]
)
