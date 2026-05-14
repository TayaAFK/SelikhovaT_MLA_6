from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_success():
    payload = {
        "gender": 1.0, "age": 27.0, "sleep_duration": 6.1,
        "quality_of_sleep": 6.0, "physical_activity": 42.0,
        "stress_level": 6.0, "bmi_category": 2.0,
        "systolic_bp": 126.0, "diastolic_bp": 83.0,
        "heart_rate": 77.0, "daily_steps": 4200.0
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code != 200:
        print(response.json().get("detail"))
        
    assert response.status_code == 200

def test_predict_invalid_data():
    payload = {"age": "dddddddddd"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
