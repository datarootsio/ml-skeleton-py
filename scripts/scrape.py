import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import settings
import subprocess
import datetime

if __name__ == '__main__':
    output = os.path.join(settings.DATA_RAW,
                          'scraped_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    subprocess.call('scrapy crawl rss --output={} -t csv --logfile=tmp/scrapy.log'
                    .format(output)
                    .split())
