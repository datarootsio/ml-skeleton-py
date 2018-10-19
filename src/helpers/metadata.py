"""Metadata helper functions."""

import json
import os
import sys
import subprocess
import datetime


def get_git_commit():
    """Get git commit if it exists.

    :return: git commit string or 'na'
    """
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode(sys.stdout.encoding).strip()

    except subprocess.CalledProcessError:
        return 'na'


def base_metadata():
    """
    Generate basic metadata.

    Includes git commit hash and a timestamp.
    BTW, subprocess.check_output is used for backwards compatibility.

    :return: metadata dictionary
    """
    return {'git_commit': get_git_commit(),
            'timestamp': str(datetime.datetime.now())}


def metadata_to_file(path, filename, metadata, logger=None):
    """Save metadata for e.g. trained models.

    Note, there should be type checking here (and elsewhere): figure out way
    to do this in a clean py2/3 compat mode.

    :param path: path were to save json file
    :param filename: filename without '.json' extension
    :param metadata: a dictionary containing the metadata (will be converted
        to json structure)
    :param logger: will also log to logger (INFO) if supplied
    """
    assert isinstance(metadata, dict)

    # TODO: this could have unwanted side effects: fix
    # bleh to backwards compat.. we need py3!
    metadata.update(base_metadata())

    # add info specific to sklearn models
    if metadata['sklearn_object'] is not None:
        metadata['model_parameters'] = metadata['sklearn_object'].get_params()
    del metadata['sklearn_object']

    fn = os.path.join(path, '{}.json'.format(filename))
    with open(fn, 'w') as f:
        json.dump(metadata, f, indent=4)

    if logger:
        logger.info(
            'metadata stored at {}: {}'.format(fn, json.dumps(metadata)))
