"""Package init.

Specify logger and load dotenv.
"""

import os
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'WARNING'))

logging \
    .getLogger(__name__) \
    .addHandler(logging.NullHandler())
