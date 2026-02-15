import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class NIDSModel:
    def __init__(self, n_estimators=150):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True

    def predict_with_confidence(self, sample):
        if not self.trained:
            raise RuntimeError("Model not trained")

        sample_scaled = self.scaler.transform(sample)
        prediction = int(self.model.predict(sample_scaled)[0])
        confidence = np.max(self.model.predict_proba(sample_scaled)) * 100

        return prediction, round(confidence, 2)
