import os
import sys

# Ajouter la racine du repo au PATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_flask_app():
    from api.app import app
    assert app is not None





# import requests

# url = "http://127.0.0.1:5000/predict"
# files = {"image": open("tests/sample_fire.jpg", "rb")}
# response = requests.post(url, files=files)
# print(response.json())