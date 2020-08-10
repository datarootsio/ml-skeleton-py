#!/usr/bin/env python3

import os
import sys

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import etl


@click.command()
def generate():
    etl.generate()


if __name__ == "__main__":
    generate()
