import os

def test_project_structure():
    assert os.path.isdir("api")
    assert os.path.isdir("model")
