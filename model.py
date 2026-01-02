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

    def predict(self, sample):
        if not self.trained:
            raise RuntimeError("Model not trained")

        sample_scaled = self.scaler.transform(sample)
        return int(self.model.predict(sample_scaled)[0])

    def evaluate(self, X_test, y_test):
        if not self.trained:
            raise RuntimeError("Model not trained")

        X_scaled = self.scaler.transform(X_test)
        preds = self.model.predict(X_scaled)
        return confusion_matrix(y_test, preds)
