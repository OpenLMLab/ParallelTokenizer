import requests


def download_file(url, filename):
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as f:
        f.write(response.content)
