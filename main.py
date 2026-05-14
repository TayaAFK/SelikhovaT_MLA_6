import joblib
import pandas as pd
import traceback
from fastapi import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time
from sqlalchemy.exc import OperationalError

DB_URL = "postgresql://user:password@db:5432/students_db"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    gender = Column(Float)
    age = Column(Float)
    sleep_duration = Column(Float)
    quality_of_sleep = Column(Float)
    physical_activity = Column(Float)
    stress_level = Column(Float)
    bmi_category = Column(Float)
    systolic_bp = Column(Float)
    diastolic_bp = Column(Float)
    heart_rate = Column(Float)
    daily_steps = Column(Float)
    prediction = Column(Float)

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class SleepData(BaseModel):
    gender: float
    age: float
    sleep_duration: float
    quality_of_sleep: float
    physical_activity: float
    stress_level: float
    bmi_category: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    daily_steps: float

def featurize(data: SleepData):
    raw_df = pd.DataFrame([data.dict()])
    
    raw_df["occupation"] = 5.0 
    
    mapping = {
        "gender": "Gender",
        "age": "Age",
        "occupation": "Occupation",
        "sleep_duration": "Sleep Duration",
        "quality_of_sleep": "Quality of Sleep",
        "physical_activity": "Physical Activity Level",
        "stress_level": "Stress Level",
        "bmi_category": "BMI Category",
        "heart_rate": "Heart Rate",
        "daily_steps": "Daily Steps",
        "systolic_bp": "Systolic_BP",
        "diastolic_bp": "Diastolic_BP"
    }
    df = raw_df.rename(columns=mapping)
    
    cols_order = [
        "Gender", "Age", "Occupation", "Sleep Duration", 
        "Quality of Sleep", "Physical Activity Level", "Stress Level", 
        "BMI Category", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"
    ]
    
    df = df[cols_order]
    
    return scaler.transform(df)

@app.on_event("startup")
def startup():
    for _ in range(5):
        try:
            Base.metadata.create_all(bind=engine)
            break
        except OperationalError:
            time.sleep(2)

@app.post("/predict")
def predict(data: SleepData):
    try:
        features = featurize(data)
        prediction = model.predict(features)[0]
        
        db = SessionLocal()
        log = PredictionLog(**data.dict(), prediction=float(prediction))
        db.add(log)
        db.commit()
        db.refresh(log)
        db.close()
        
        return {"predicted_disorder_code": int(prediction)}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{error_details}")
        raise HTTPException(status_code=500, detail=error_details)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
