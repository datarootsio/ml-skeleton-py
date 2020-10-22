from setuptools import setup, find_packages


test_deps = [
    "pytest>=5.3.5",
    "pytest-flask>=1.0.0",
    "pip>=20.0.0",
    "tox>=3.14.0",
    "flake8>=3.7.9",
    "flake8-annotations>=1.1.3",
    "pytest-cov>=2.8.1",
    "black>=19.10b0"
]

serve_deps = [
    "dploy-kickstart>=0.1.5",
]

extras = {"test": test_deps, "serve": serve_deps}

setup(
    name="ml-skeleton-py",
    version="0.1.0",
    url="dataroots.io",
    author="dataroots.io",
    author_email="info@dataroots.io",
    description="Description of my ml-skeleton package",
    packages=find_packages(),
    install_requires=["pandas>=1.1.0", "scikit-learn>=0.23.2"],
    tests_require=test_deps,
    extras_require=extras,
)
