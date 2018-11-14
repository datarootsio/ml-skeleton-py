"""Example of calling API endpoint."""

import requests
from src import settings as s

if __name__ == '__main__':

    input_json = {
        "features": [
            [4.8, 1.8, 2.4],
            [7.4, 2.5, 3.1]
        ]
    }

    response = requests.post("http://{}:{}/predict"
                             .format(s.FLASK_ENDPOINT_HOST,
                                     s.FLASK_ENDPOINT_PORT),
                             json=input_json)
    print(response.json())
