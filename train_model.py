import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

def train():
    df = pd.read_csv("./df_clear.csv")
    
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SGDClassifier(random_state=42, loss='log_loss')
    model.fit(X_scaled, y)
    
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

if __name__ == "__main__":
    train()
