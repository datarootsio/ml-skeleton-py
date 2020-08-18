"""Package init.

Specify logger and load dotenv.
"""

import os
import logging
from dotenv import load_dotenv, find_dotenv
from dynaconf import Dynaconf

load_dotenv(find_dotenv())

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

logging.getLogger(__name__).addHandler(logging.NullHandler())


settings = Dynaconf(
    # export envvars with `export DYNACONF_FOO=bar`.
    envvar_prefix="DYNACONF",
    # Load files in the given order.
    settings_files=["settings.yaml", ".secrets.yaml"],
)
