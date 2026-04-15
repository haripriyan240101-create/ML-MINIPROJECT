import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ================== LOAD DATA ==================
data = pd.read_csv("stress_dataset.csv")

# ================== ENCODE ==================
le_stress = LabelEncoder()
data['stress_level'] = le_stress.fit_transform(data['stress_level'])

# ================== FEATURES ==================
X = data[['study_hours','sleep_hours','social_hours']]
y = data['stress_level']

# ================== MODEL ==================
model = RandomForestClassifier()
model.fit(X, y)

# ================== METRICS ==================
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

# ================== STATIC FOLDER ==================
if not os.path.exists("static"):
    os.makedirs("static")

# ================== METRICS GRAPH ==================
def create_metrics_graph():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]

    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0,1)
    plt.title("Model Performance Metrics")
    plt.savefig("static/metrics.png")
    plt.close()

create_metrics_graph()

# ================== ROUTES ==================
@app.route('/')
def home():
    return render_template("index.html",
                           result=None,
                           confidence=None,
                           suggestion=None,
                           study=None,
                           sleep=None,
                           social=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        study = int(request.form.get('study'))
        sleep = int(request.form.get('sleep'))
        social = int(request.form.get('social'))

        # Prediction
        prediction = model.predict([[study, sleep, social]])
        probs = model.predict_proba([[study, sleep, social]])

        result = le_stress.inverse_transform(prediction)[0]
        confidence = round(max(probs[0]) * 100, 2)

        # Suggestions
        if result == 'high':
            suggestion = "⚠️ Reduce workload and get more rest."
        elif result == 'moderate':
            suggestion = "🙂 Maintain balance."
        else:
            suggestion = "✅ Good routine!"

        # Dynamic graph
        labels = ['Low', 'Moderate', 'High']
        values = [0, 0, 0]

        if result == 'low':
            values = [1, 0, 0]
        elif result == 'moderate':
            values = [0, 1, 0]
        else:
            values = [0, 0, 1]

        plt.figure()
        plt.bar(labels, values)
        plt.title("Predicted Stress Level")
        plt.savefig("static/graph.png")
        plt.close()

        # Feature importance
        importance = model.feature_importances_
        features = ['Study Hours', 'Sleep Hours', 'Social Hours']

        plt.figure()
        plt.bar(features, importance)
        plt.title("Feature Importance")
        plt.savefig("static/feature.png")
        plt.close()

        return render_template("index.html",
                               result=result,
                               confidence=confidence,
                               suggestion=suggestion,
                               study=study,
                               sleep=sleep,
                               social=social)

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html",
                               result=None,
                               confidence=None,
                               suggestion=None,
                               study=None,
                               sleep=None,
                               social=None)


# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)
