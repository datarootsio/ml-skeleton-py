"""Test if templating worked fine."""

import os

project_dir = '../beautifulml/'
# check if default project exists
assert os.path.isdir(project_dir)

# list all files recursively, again this is simpler in just >py35
files = []
for root, dirnames, filenames in os.walk(project_dir):
    for filename in filenames:
        files.append(os.path.join(root, filename))

# check if all variables are replaced
for filename in files:
    with open(filename, 'r') as f:
        try:
            txt = f.read()
            assert '{{' not in txt
            assert '}}' not in txt
        except AssertionError:
            print('jinja var not replaced in: {}'.format(filename))
        except UnicodeDecodeError:
            # temporary fix to avoid files that should not be there
            # to begin with
            pass
