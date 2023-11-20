import requests
import sys
import json

GDRIVE_URL = "https://docs.google.com/uc?export=download"
CHUNK_SIZE = 32768

# Thanks to https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_from_gdrive(id: str, filename: str):
    session = requests.Session()

    params = { "id": id, "confirm": 1}
    resp = session.get(GDRIVE_URL, params=params, stream=True)
    token = get_confirm_token(resp)

    if token:
        params = { "id": id, "confirm": token }
        resp = session.get(GDRIVE_URL, params=params, stream=True)

    with open(filename, "wb") as writer:
        for chunk in resp.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                writer.write(chunk)

def get_confirm_token(resp):
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None

if __name__ == "__main__":
    download_from_gdrive("1Z4ZGofzgGPFkV8hSEuaG8xMiD3PnzEX4", "yolo.zip")
