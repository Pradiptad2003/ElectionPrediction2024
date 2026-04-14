from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import joblib  # For loading the machine learning model
from matplotlib.figure import Figure
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load MLP regression model (ensure it's compatible for loading)
model = joblib.load('/mnt/data/mlp_regression.ipynb')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_regression', methods=['GET', 'POST'])
def predict_regression():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            data = pd.read_csv(filepath)
            
            # Predict using the model (modify if necessary)
            predictions = model.predict(data)
            
            # Generate regression plot
            fig = Figure()
            ax = fig.subplots()
            ax.plot(data, predictions, label='Regression Line')
            ax.legend()
            ax.set_title('Election Prediction using Regression')
            
            # Convert plot to displayable image format
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            return render_template('prediction_result.html', prediction_image=img_data, predictions=predictions)

    return render_template('predict_regression.html')

@app.route('/predict_sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            data = pd.read_csv(filepath)
            
            # Placeholder for sentiment analysis code
            sentiments = sentiment_analysis(data)
            
            return render_template('prediction_result.html', sentiments=sentiments)

    return render_template('predict_sentiment.html')

def sentiment_analysis(data):
    # Placeholder for actual sentiment analysis implementation
    return ["Positive" if row[0] > 0.5 else "Negative" for row in data.values]

if __name__ == '__main__':
    app.run(debug=True)
