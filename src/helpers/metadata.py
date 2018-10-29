"""Metadata helper functions."""
import json
import os
import sklearn
import sys
import subprocess
import datetime
import hashlib

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


def generate_metadata(model_location, model_description, model_object,
                      data_location, data_identifier, features_object,
                      testing_strategy, scores, model_hyperparameters=None,
                      extra_metadata=None):
    """
    Generate metadata for provided information.

    More detailed documentation can be found in HOWTO.

    :param model_location: Path to saved serialized model.
    :param model_description: Textual description of the model (free-form).
    :param model_object: Model object.
    :param data_location: Path to data used for model training.
    :param data_identifier: Uniquely identify data version used for training
    the model. Useful for model reproducibility. Can be None if it is not
    possible to have such identifier.
    :param features_object: Object (dataframe) representing train data
    (only features, no labels).
    :param testing_strategy: Description of the strategy used for
    testing/evaluating model.
    :param scores: Model evaluation results. Refer to HOWTO for more details.
    :param model_hyperparameters: Map (key-value pairs) containing
    hyperparameters used when model was fitted.
    :param extra_metadata: Additional metadata user can provide as a map.
    :return: metadata object
    """
    if model_object is None:
        raise Exception('Model object must be provided for metadata.')
    if features_object is None:
        raise Exception('Features object must be provided for metadata.')

    if check_scores_structure(scores) is False:
        raise Exception('Metada format for \'scores\' field is not valid.')

    # Extract several metadata...

    model_type = str(model_object.__module__
                     + "." + model_object.__class__.__name__)

    model_id = hashlib.sha1(
        str(get_git_commit()).encode('utf-8') +
        str(datetime.datetime.now()).encode('utf-8')) \
        .hexdigest()

    feature_names = list(features_object.columns.values)
    num_data_rows = features_object.shape[0]
    num_data_features = features_object.shape[1]

    if model_hyperparameters is None:
        if isinstance(model_object, sklearn.base.BaseEstimator):
            model_hyperparameters = model_object.get_params()

    model_object_size = '{} bytes'.format(sys.getsizeof(model_object))
    data_object_size = '{} bytes'.format(sys.getsizeof(features_object))

    metadata = {
        'model_location': model_location,
        'model_description': model_description,
        'model_identifier': model_id,
        'model_type': model_type,
        'model_hyperparameters': model_hyperparameters,
        'model_size': model_object_size,
        'input_data_location': data_location,
        'input_data_identifier': data_identifier,
        'num_data_rows': num_data_rows,
        'num_data_features': num_data_features,
        'feature_names': feature_names,
        'data_size': data_object_size,
        'testing_strategy': testing_strategy,
        'scores': scores
    }

    if extra_metadata is not None:
        metadata.update(extra_metadata)

    # TODO: this could have unwanted side effects: fix
    # bleh to backwards compat.. we need py3!
    metadata.update(base_metadata())

    return metadata


def save_metadata(path, filename, metadata, logger=None):
    """
    Save metadata for trained models.

    :param path: path were to save json file
    :param filename: filename without '.json' extension
    :param metadata: a dictionary containing the metadata (will be converted
        to json structure)
    :param logger: will also log to logger (INFO) if supplied
    """
    assert isinstance(metadata, dict)

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
