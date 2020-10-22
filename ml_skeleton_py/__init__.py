"""Package init.

Specify logger
"""

import os
import logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logging.getLogger(__name__).addHandler(logging.NullHandler())
