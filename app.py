from flask import Flask, render_template, request
import os
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from mlp_regression import predict_with_mlp

#  Headless server issue (EC2)
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Regression route
@app.route('/predict_regression', methods=['GET', 'POST'])
def predict_regression():
    if request.method == 'POST':
        file = request.files.get('file')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                predictions, plot_path = predict_with_mlp(filepath)

                with open(plot_path, "rb") as image_file:
                    img_data = base64.b64encode(image_file.read()).decode("utf-8")

                return render_template(
                    'prediction_result.html',
                    prediction_image=img_data,
                    predictions=predictions.tolist()
                )

            except Exception as e:
                return render_template('predict_regression.html', error=str(e))

    return render_template('predict_regression.html')


# Sentiment route
@app.route('/predict_sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        file = request.files.get('file')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            plot_path, r_value, p_value, hypothesis_result = sentiment_analysis(filepath)

            with open(plot_path, "rb") as image_file:
                img_data = base64.b64encode(image_file.read()).decode("utf-8")

            return render_template(
                'prediction_result.html',
                prediction_image=img_data,
                r_value=r_value,
                p_value=p_value,
                result=hypothesis_result
            )

    return render_template('predict_sentiment.html')


# Sentiment analysis function
def sentiment_analysis(data_path):
    data = pd.read_csv(data_path)

    sentiment_scores = data['Sentiment Score']

    # column name corrected (removed extra space)
    vote_shares = data['Vote']

    parties = data['Party']

    plt.figure(figsize=(8, 6))

    plt.scatter(sentiment_scores, vote_shares, s=100, edgecolors='black')

    for sentiment, vote, party in zip(sentiment_scores, vote_shares, parties):
        plt.text(sentiment, vote, party, fontsize=10)

    # Trend line
    m, b = np.polyfit(sentiment_scores, vote_shares, 1)
    trend_line = np.array(sentiment_scores) * m + b
    plt.plot(sentiment_scores, trend_line, linestyle="dashed", color="black")

    # Correlation
    r_value, p_value = pearsonr(sentiment_scores, vote_shares)

    alpha = 0.05
    if p_value < alpha:
        hypothesis_result = "Significant correlation exists."
    else:
        hypothesis_result = "No significant correlation."

    plt.xlabel("Sentiment Score")
    plt.ylabel("Vote Share")
    plt.title("Sentiment vs Vote Share")
    plt.grid(True)

    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path, round(r_value, 4), round(p_value, 4), hypothesis_result


# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
