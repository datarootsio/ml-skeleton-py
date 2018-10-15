"""Setup configuration.

installation config
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-skeleton-py",
    version="0.0.1",
    author="dataroots",
    author_email="info@dataroots.io",
    description="A Python ML framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()
)
