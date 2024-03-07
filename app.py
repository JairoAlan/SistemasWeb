from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Cargamos los datos
data = pd.read_csv('diabetes.csv')

# Separar las características en (X) y la variable objetivo en (y)
X = data.drop('Outcome', axis=1)  # Características
y = data['Outcome']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características para asegurar que todas tengan la misma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", logistic_result=None, random_forest_result=None)

@app.route("/predict", methods=["POST"])
def predict():
    glucose = float(request.form["glucose"])
    bmi = float(request.form["bmi"])
    age = int(request.form["age"])

    additional_features = [0, 0, 0, 0, 0]

    features = [[glucose, bmi, age] + additional_features]
    scaled_features = scaler.transform(features)

    logistic_prediction = logistic_model.predict(scaled_features)
    

    logistic_result = "Diabético" if logistic_prediction[0] == 1 else "No Diabético"
    

    return render_template("index.html", logistic_result=logistic_result)

if __name__ == "__main__":
    app.run(debug=True)
