from api.app import app

def test_flask_app_exists():
    assert app is not None





# import requests

# url = "http://127.0.0.1:5000/predict"
# files = {"image": open("tests/sample_fire.jpg", "rb")}
# response = requests.post(url, files=files)
# print(response.json())