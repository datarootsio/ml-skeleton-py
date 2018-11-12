"""Example how to start Flask endpoint, to serve the predictions."""
import os
import sys

import click
import logging
from mlmonkey.api import create_flask_app
# We need to do absolute import because of the Flask.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src  import settings as s

logger = logging.getLogger(__name__)


@click.command()
@click.option('--model-path',
              default='{}/model.p'.format(s.MODEL_DIR))
# For now this is fixed, because with issues with Flask relative imports.
@click.option('--host', default=s.FLASK_ENDPOINT_HOST)
@click.option('--port', default=s.FLASK_ENDPOINT_PORT)
def main(host, port, model_path):
    """Load model and create flask app.

    :param model_path: path to serialized model
    :param host: server host
    :param port: port
    """
    app = create_flask_app(model_path)
    logger.info('Starting flask server...')
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()
