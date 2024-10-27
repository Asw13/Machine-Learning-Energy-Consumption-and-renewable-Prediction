import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your pre-trained model
model_path = "model/lstm_model.pkl"  # Replace with your actual model path
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file1' not in request.files or 'file2' not in request.files:
        return "Both files must be provided."

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Read uploaded CSV file
    data = pd.read_csv(file1)
    data1=pd.read_csv(file2)
    # Check if the columns match the expected columns
    expected_columns = ['IRR (W/m2)', 'MODULE_TEMP', 'Amb_Temp', 'AC Power in Watts']
    if not data.columns.equals(pd.Index(expected_columns)):
        return "Invalid column names in uploaded file.", 400

    else :
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        data2=data1.values
        def create_sequences(data, sequence_length,data2):
            X = []
            y = []
            Z=[]
            for i in range(len(data) - sequence_length):
                X.append(data[i:i + sequence_length, :-1])  # Excluding target from inputs
                y.append(data[i + sequence_length, -1])
                  # Target (AC Power in Watt, for example)
            for i in range(len(data2) - sequence_length):
                Z.append(data2[i:i + sequence_length])
            return np.array(X), np.array(y),np.array(Z)
        
        
        sequence_length = 60  # Adjust based on time dependency
        X, y ,Z= create_sequences(scaled_data, sequence_length,data2)
        # Train-test split


    # Process and prepare the data as needed
    # For example, if scaling is necessary, apply it here

    # Ensure data matches model's input shape
    # Example: Predicting next 10 days
    model.fit(X,y)

    future_data = model.predict(Z[-10:])
    future_data_expanded = np.hstack([future_data, np.zeros((10, 3))])
    predictions = scaler.inverse_transform(future_data_expanded)
    predictions = predictions[:, 0]


    # Convert predictions to a DataFrame for easy display
    predictions_df = pd.DataFrame(predictions, columns=["Predicted Value"])

        # Create a plot of the predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_df.index, predictions_df['Predicted Value'], marker='o', color='b', label='Predicted Value')
    plt.title('Predicted AC Power in Watts')
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted Value (W)')
    plt.grid()
    plt.legend()
    plot_path = os.path.join('static', 'predictions_plot.png')  # Path to save the plot
    # Ensure the directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

    # Render predictions on a new page or return as a response
    return render_template('results.html', tables=[predictions_df.to_html(classes='data')], titles=predictions_df.columns.values,plot_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)

