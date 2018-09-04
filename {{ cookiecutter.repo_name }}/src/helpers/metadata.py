import json
import os

# note there should be type checking here: figure out way to do this in a py2/3 compat mode
def save_metadata(path, filename, metadata):
    """
    Save metadata for e.g. trained models
    :param path: path were to save json file
    :param filename: filename without '.json' extension
    :param metadata: a dictionary containing the metadata (will be converted to json structure)
    """
    assert isinstance(metadata, dict)

    with open(os.path.join(path, '{}.json'.format(filename)), 'w') as f:
        json.dump(metadata, f)