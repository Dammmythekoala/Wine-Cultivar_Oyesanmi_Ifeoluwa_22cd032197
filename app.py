from flask import Flask, render_template, request
import joblib
import numpy as np
import os
app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl'))
FEATURES = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'total_phenols', 'flavanoids']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        vals = [float(request.form.get(f)) for f in FEATURES]
    except Exception as e:
        return f'Invalid input: {e}', 400
    arr = np.array(vals).reshape(1, -1)
    pred = model.predict(arr)[0] + 1  # map 0/1/2 to 1/2/3
    return render_template('index.html', features=FEATURES, result=f'Predicted cultivar: Cultivar {pred}', inputs=dict(zip(FEATURES, vals)))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
