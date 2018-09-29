"""Setup configuration.

installation config
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="{{ cookiecutter.repo_name }}",
    version="0.0.1",
    author="{{ cookiecutter.company }}",
    author_email="{{ cookiecutter.company_email }}",
    description="{{ cookiecutter.project_short_description }}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()
)
