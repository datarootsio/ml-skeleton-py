import os
import logging
import shutil

logger = logging.getLogger(__name__)

def main(model_name):
    print(model_name)
    logger.info('Copying iris dataset from raw to tranformed')
    logger.info('Skipping staging, not relevant for this example')

    shutil.copyfile(
        os.path.join(os.getenv('DATA_RAW'), 'iris.csv'),
        os.path.join(os.getenv('DATA_TRANSFORMED'), 'iris.csv')
    )

    logger.info('Done')

if __name__ == '__main__':
    main()
