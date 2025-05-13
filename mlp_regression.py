import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def predict_with_mlp(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data['%Vote'].values  # Features
    y = data["Seat Share"].values   # Target variable

    # Initialize and train the MLP regressor
    model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=1)
    model.fit(X.reshape(-1,1), y)

    # Make predictions
    predictions = model.predict(X.reshape(-1,1))

    # Create a plot showing actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)  # Scatter plot of actual data
    plt.scatter(X, predictions, color='red', label='Predicted Data', alpha=0.6)  # Scatter plot of predicted data
    plt.plot(X, predictions, color='green', linestyle='--', label='Regression Line')  # Regression line
    plt.legend()
    plt.xlabel('% Vote Share')
    plt.ylabel('Seat Share')
    plt.title('MLP Regression Prediction %Vote Share vs Seat Share')

    # Save the plot
    plot_path = 'static/regression_plot.png'
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid memory leaks

    return predictions, plot_path
