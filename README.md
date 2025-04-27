# Machine Learning Model Training and Prediction Web App

## Project Overview

This project is a **Streamlit-based web application** designed to assist users in performing machine learning tasks, including model training, prediction, and visualization. The app supports the following features:

- **Model Training**: Users can select and train various machine learning models (Logistic Regression, Random Forest, SVM, XGBoost) on uploaded datasets. The trained models can be saved and later used for making predictions.
  
- **Model Prediction**: After training a model, users can upload new datasets (for prediction) and get predictions using the selected pre-trained model. The app will display the predicted labels for the new data based on the trained model.

- **Model Evaluation**: The app provides a **confusion matrix** for evaluating the performance of the model on the uploaded data. It visualizes the true vs. predicted labels for each class, helping users understand how well the model is performing.

- **Accuracy Calculation**: The app calculates and displays the **accuracy** of the model on the uploaded dataset. Accuracy is computed as the percentage of correct predictions made by the model.

- **Data Drift Detection (Optional)**: The app allows users to detect data drift in their datasets by comparing the characteristics of the current dataset with historical data (if available). This feature helps in identifying changes in the data distribution that could affect model performance.

- **Visualization**: The app provides visualizations for model performance, including a **confusion matrix** and **accuracy** metrics, which help users assess the effectiveness of their models.

In summary, the project provides a complete **end-to-end solution** for machine learning tasks, from training models to evaluating their performance and making predictions on new data.

---

## Features

- **Model Training**: Train models using uploaded datasets and select from multiple machine learning algorithms (Logistic Regression, Random Forest, SVM, XGBoost).
  
- **Prediction**: Upload new data for prediction and get results based on the trained model.

- **Confusion Matrix**: Visualize the confusion matrix of the model's predictions on the test data.

- **Model Accuracy**: Display the accuracy of the model based on the predictions on the uploaded data.

- **Data Drift Detection**: Optional feature to compare the uploaded data with historical data to check for data drift.

---
# My ML Project

This project includes machine learning model training, predictions, and visualization. Below is the layout with images and instructions.

<img src="src/img1.png" alt="Resized Example Image" height="400" width="200">
 <!-- Text (Commands) in the second column -->
  <div>
    <h3>Installation and Setup:</h3>
    <pre>
    pip install -r requirements.txt
    </pre>
  </div>

  <div>
    <h3>Running the App:</h3>
    <pre>
    streamlit run app.py
    </pre>
  </div>

</div>
.


