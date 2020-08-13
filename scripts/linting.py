import os, sys
import subprocess


def code_quality_score():
    """
	code_quality_score computes the code quality in the repo
	"""

    print("Computing code quality...")

    flake8_errors = subprocess.check_output("flake8 .. | wc -l", shell=True)
    print("This repo has {} flake8 errors.".format(int(flake8_errors)))
    pydoc_errors = subprocess.check_output("pydocstyle .. | wc -l", shell=True)
    print("This repo has {} pydoc errors.".format(int(pydoc_errors)))
    mypy_errors = subprocess.check_output("mypy .. | wc -l", shell=True)
    print("This repo has {} mypy errors.".format(int(mypy_errors)))
    lines_of_code = subprocess.check_output(
        "pygount --suffix py | awk '{sum += $1} END {print sum}'", shell=True
    )
    print("This repo has {} lines of code.".format(int(lines_of_code)))

    print("Done!")
    return


if __name__ == "__main__":
    code_quality_score()
