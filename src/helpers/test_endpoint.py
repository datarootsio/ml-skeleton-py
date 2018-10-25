"""Example of calling API endpoint."""

import requests

if __name__ == '__main__':

    input_json = {
        "features": [
            [5.8, 2.8, 2.4],
            [6.4, 2.8, 2.1]
        ]
    }

    response = requests.post("{}/predict"
                             .format('http://localhost:5000'),
                             json=input_json)
    print(response.json())
