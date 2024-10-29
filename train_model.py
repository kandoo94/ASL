# train_model.py

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    with open("dataset/preprocessed_data.pkl", "rb") as f:
        data, labels = pickle.load(f)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    
    with open("models/asl_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved.")

if __name__ == "__main__":
    train_model()
