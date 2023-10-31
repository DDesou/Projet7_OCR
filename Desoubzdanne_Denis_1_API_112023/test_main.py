from fastapi.testclient import TestClient
from fastapi import status
from main import app

client = TestClient(app=app)

def test_index_returns_correct():
    response = client.get('/')

    assert response.status_code == status.HTTP_200_ok
    assert response.json() == {'message' : 'my API works!!'}