import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

class CryptoPriceModel:
    def __init__(self, window_size, input_features):
        """
        Initialisation du modèle.

        Parameters:
        - window_size: nombre de pas de temps pour les séquences.
        - input_features: nombre de caractéristiques d'entrée.
        """
        self.window_size = window_size
        self.input_features = input_features
        self.model = self._build_model()

    def _build_model(self):
        """
        Construit le modèle de réseau de neurones.

        Returns:
        - Un modèle compilé.
        """
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, input_shape=(self.window_size, self.input_features)),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True, input_shape=(self.window_size, self.input_features)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X_train, y_train, validation_data=None, epochs=20, batch_size=32):
        """
        Entraîne le modèle.

        Parameters:
        - X_train: données d'entraînement (features).
        - y_train: données d'entraînement (target).
        - validation_data: tuple (X_val, y_val) pour validation.
        - epochs: nombre d'époques.
        - batch_size: taille du batch.

        Returns:
        - L'historique d'entraînement.
        """
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle.

        Parameters:
        - X_test: données de test (features).
        - y_test: données de test (target).

        Returns:
        - La perte et l'erreur moyenne absolue.
        """
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        """
        Fait des prédictions.

        Parameters:
        - X: données d'entrée.

        Returns:
        - Les prédictions du modèle.
        """
        return self.model.predict(X)

    def plot_predictions(self, y_true, y_pred, title="Predictions vs Real Prices"):
        """
        Affiche les prédictions comparées aux vraies valeurs.

        Parameters:
        - y_true: valeurs réelles.
        - y_pred: prédictions.
        - title: titre du graphique.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="True Prices")
        plt.plot(y_pred, label="Predicted Prices")
        plt.legend()
        plt.title(title)
        plt.show()

    def save_model(self, filepath):
        """
        Sauvegarde le modèle sur disque.

        Parameters:
        - filepath: chemin du fichier pour sauvegarder le modèle.
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Charge un modèle depuis le disque.

        Parameters:
        - filepath: chemin du fichier pour charger le modèle.
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
