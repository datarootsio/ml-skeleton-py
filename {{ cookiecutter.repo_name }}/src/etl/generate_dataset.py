import os
import logging
import shutil

logger = logging.getLogger(__name__)

def main():
    logger.info('Copying iris dataset from raw to tranformed')

    shutil.copyfile(
        os.path.join(os.getenv('DATA_RAW'), 'iris.csv'),
        os.path.join(os.getenv('DATA_TRANSFORMED'), 'iris.csv')
    )

    logger.info('Done')

if __name__ == '__main__':
    main()
