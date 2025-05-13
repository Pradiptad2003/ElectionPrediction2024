from flask import Flask, render_template, request
import os
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from mlp_regression import predict_with_mlp  # Import the function for MLP prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for regression prediction
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
                
                return render_template('prediction_result.html', prediction_image=img_data, predictions=predictions.tolist())

            except ValueError as e:
                return render_template('predict_regression.html', error=str(e))

    return render_template('predict_regression.html')

# Route for sentiment analysis prediction
@app.route('/predict_sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Perform sentiment analysis and get results
            plot_path, r_value, p_value, hypothesis_result = sentiment_analysis(filepath)
            
            # Convert plot image to base64 for rendering in HTML
            with open(plot_path, "rb") as image_file:
                img_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Render results on the webpage
            return render_template('prediction_result.html', 
                                   prediction_image=img_data)

    return render_template('predict_sentiment.html')

# Function for sentiment analysis
def sentiment_analysis(data_path):
    # Load dataset
    data = pd.read_csv(data_path)
    sentiment_scores = data['Sentiment Score']
    vote_shares = data['Vote ']

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(sentiment_scores, vote_shares, color=['Orange', 'Green', 'Blue', 'Red'], s=100, edgecolors='black')

    for sentiment, vote, party in zip(sentiment_scores, vote_shares, data['Party']):
        plt.text(sentiment + 5, vote, party, fontsize=12)

    # Fit a trend line (Linear Regression)
    m, b = np.polyfit(sentiment_scores, vote_shares, 1)
    trend_line = np.array(sentiment_scores) * m + b
    plt.plot(sentiment_scores, trend_line, linestyle="dashed", color="black", label="Trend Line")

    # Calculate Pearson correlation coefficient and p-value
    r_value, p_value = pearsonr(sentiment_scores, vote_shares)

    # Hypothesis testing
    alpha = 0.05
    if p_value < alpha:
        hypothesis_result = "There is a significant correlation between sentiment score and vote share."
    else:
        hypothesis_result = "There is no significant correlation between sentiment score and vote share."
    '''''
    # Add correlation and p-value text on the right side of the plot
    plt.text(
        max(sentiment_scores) * 0.9, max(vote_shares) * 0.9,
        f"r = {r_value:.4f}\np = {p_value:.4f}",
        fontsize=12, bbox=dict(facecolor='white', alpha=0.5), ha='right'
    )
'''
    plt.xlabel("Sentiment Score (S)")
    plt.ylabel("Vote Share (V)")
    plt.title("Sentiment Score vs Vote Share")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path, round(r_value, 4), round(p_value, 4), hypothesis_result


if __name__ == '__main__':
    app.run(debug=True)