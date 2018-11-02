"""Dataset generation example."""

import os
import logging
import shutil

from .. import settings as s

logger = logging.getLogger(__name__)


def generate():
    """Generate a dataset using raw input data.

    :param model_name: the name of the
    """
    logger.info('Copying iris dataset from raw to tranformed')
    logger.info('Skipping staging, not relevant for this example')

    shutil.copyfile(
        os.path.join(s.DATA_RAW, 'iris.csv'),
        os.path.join(s.DATA_TRANSFORMED, 'iris.csv')
    )

    logger.info('Done')
