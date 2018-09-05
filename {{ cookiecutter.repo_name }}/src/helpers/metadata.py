import json
import os
import sys
import subprocess
import datetime

def base_metadata():
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(sys.stdout.encoding).strip(),
     'timestamp': str(datetime.datetime.now())}

# note there should be type checking here: figure out way to do this in a py2/3 compat mode
# subprocess.check_output used for backwards compat
def metadata_to_file(path, filename, metadata, logger=None):
    """
    Save metadata for e.g. trained models
    :param path: path were to save json file
    :param filename: filename without '.json' extension
    :param metadata: a dictionary containing the metadata (will be converted to json structure)
    :param logger: will also log to logger (INFO) if supplied
    """
    assert isinstance(metadata, dict)

    # TODO: this could have unwanted side effects: fix (bleh to backwards compat)
    metadata.update(base_metadata())

    fn = os.path.join(path, '{}.json'.format(filename))
    with open(fn, 'w') as f:
        json.dump(metadata, f, indent=4)

    if logger:
        logger.info('metadata stored at {}: {}'.format(fn, json.dumps(metadata)))
