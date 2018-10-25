"""Metadata helper functions."""
import json
import os
import sklearn
import sys
import subprocess
import datetime
from .. import settings as s


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


def generate_metadata(path, filename, metadata, logger=None):
    """
    Save metadata for e.g. trained models.

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
        metadata['model_hyperparameters'] = \
            metadata['sklearn_object'].get_params()
    del metadata['sklearn_object']

    if check_scores_structure(metadata['scores']) is False:
        raise Exception('Metada format for \'scores\' field is not valid.')

    fn = os.path.join(path, '{}.json'.format(filename))
    with open(fn, 'w') as f:
        json.dump(metadata, f, indent=4)

    if logger:
        logger.info(
            'Metadata stored at {}: {}'.format(fn, json.dumps(metadata)))


def check_scores_structure(scores):
    """
    Check if scores comply to predefined structure.

    :param scores: map containing scoring data
    :return: boolean value, indicating whether the structure is valid
    """
    assert isinstance(scores, dict)

    # get all sklearn score names
    sklearn_metric_names = sorted(sklearn.metrics.SCORERS.keys())
    valid_metric_names = sklearn_metric_names + \
        ['log_loss', 'mean_absolute_error', 'mean_squared_error',
         'mean_squared_log_error', 'median_absolute_error']

    used_metrics = scores.keys()
    for metric in used_metrics:
        if (metric not in valid_metric_names and
                not metric.startswith('custom')):
            return False

        metric_scores = scores[metric]
        for score in metric_scores:
            if (score not in ['cross_val', 'hold_out'] and
                    not score.startswith('custom')):
                return False

    return True


# Deprecated method! We will keep it for now in case it is useful.
def most_recent_model_id():
    """Return id of the most recent model."""
    metadata_maps = []
    for file in os.listdir(s.MODEL_METADATA_DIR):
        if file.endswith('.json'):
            file_path = "{}/{}".format(s.MODEL_METADATA_DIR, file)
            with open(file_path, 'r') as f:
                dict = json.load(f)
                metadata_maps.append(dict)

    metadata_maps = sorted(metadata_maps,
                           key=lambda dict: dict['timestamp'],
                           reverse=True)

    if len(metadata_maps) > 0:
        return metadata_maps[0]['model_identifier']
    return None