"""Configuration file.

Most of these variables can be overridden through environment variables.
"""

import os

DATA_RAW = os.path.join(os.path.abspath(os.getenv('DATA_RAW', './data/raw/')), '')
DATA_TRANSFORMED = os.path.join(os.path.abspath(os.getenv('DATA_TRANSFORMED', './data/transformed/')), '')
DATA_STAGING = os.path.join(os.path.abspath(os.getenv('DATA_TRANSFORMED', './data/staging/')), '')
DATA_PREDICTIONS = os.path.join(os.path.abspath(os.getenv('DATA_TRANSFORMED', './data/predictions/')), '')

MODEL_DIR = os.path.join(os.path.abspath(os.getenv('MODEL_DIR', './models')), '')
MODEL_METADATA_DIR = os.path.join(os.path.abspath(os.getenv('MODEL_METADATA_DIR', './models/metadata')), '')

FLASK_ENDPOINT_HOST = os.getenv('FLASK_ENDPOINT_HOST', '0.0.0.0')
FLASK_ENDPOINT_PORT = os.getenv('FLASK_ENDPOINT_PORT', 5000)



# Scrapy settings for scrape project
#
# more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'scrape'

SPIDER_MODULES = ['src.scrape.spiders']
NEWSPIDER_MODULE = 'src.scrape.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True