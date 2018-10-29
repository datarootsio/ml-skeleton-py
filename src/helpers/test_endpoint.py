"""Example of calling API endpoint."""

import requests
from .. import settings as s

if __name__ == '__main__':

    input_json = {
        "features": [
            [5.8, 2.8, 2.4],
            [6.4, 2.8, 2.1]
        ]
    }

    response = requests.post("http://{}:{}/predict"
                             .format(s.FLASK_ENDPOINT_HOST,
                                     s.FLASK_ENDPOINT_PORT),
                             json=input_json)
    print(response.json())
