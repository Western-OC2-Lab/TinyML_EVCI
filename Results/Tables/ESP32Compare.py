import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from tensorflow.keras.models import load_model
# import numpy as np

def calculate_average_from_csv_pandas(file_path, column_name):
    df = pd.read_csv(file_path)
    return df[column_name].mean()


file_path = 'C:/Users/FDEHROUY/Desktop/Run/MicoroctrollerCompare/ML_MLP_PerSamplePredictionTime_ms.csv'
column_name = 'Time'
average = calculate_average_from_csv_pandas(file_path, column_name)
print(f"The average of {column_name} for ML_MLP is {average} ms")

# model_path = 'C:/Users/FDEHROUY/Desktop/Run/MicoroctrollerCompare/ML_MLP.h5'
# ML_MLP = load_model(model_path)
# X_test = pd.read_csv('C:/Users/FDEHROUY/Desktop/Run/MicoroctrollerCompare/X_test.csv')
y_test = pd.read_csv('C:/Users/FDEHROUY/Desktop/Run/MicoroctrollerCompare/y_test.csv')
# # calculate accuracy, precision, recall, and F1 score for the entire test set
# y_preds = ML_MLP.predict(X_test)
# y_preds = np.argmax(y_preds, axis=1)
# accuracy = accuracy_score(y_test, y_preds)
# precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
# recall = recall_score(y_test, y_preds, average='weighted')
# f1 = f1_score(y_test, y_preds, average='weighted')
# print("ML_MLP Aggregated Metrics for Accuracy, Precision, Recall, and F1 Score:")
# print(f"Accuracy: {accuracy:.6f}")
# print(f"Precision: {precision:.6f}")
# print(f"Recall: {recall:.6f}")
# print(f"F1 Score: {f1:.6f}")

file_path = 'C:/Users/FDEHROUY/Desktop/Run/MicoroctrollerCompare/TinyML_MLP_PerSamplePredictionTime_us.csv'
column_name = 'Time'
average = calculate_average_from_csv_pandas(file_path, column_name)
print(f"The average of {column_name} for TinyML_MLP is {average} us")

file_path = 'C:/Users/FDEHROUY/Desktop/Run/ESP32C/Client_CPU/Results.csv'
column_name = 'Inference_Time'
average = calculate_average_from_csv_pandas(file_path, column_name)
print(f"The average of {column_name} for ESP32 is {average} us")
results = pd.read_csv('C:/Users/FDEHROUY/Desktop/Run/ESP32C/Client_CPU/Results.csv')
y_preds = results['Model_Output']
accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
recall = recall_score(y_test, y_preds, average='weighted')
f1 = f1_score(y_test, y_preds, average='weighted')
print("ML_MLP Aggregated Metrics for Accuracy, Precision, Recall, and F1 Score:")
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 Score: {f1:.6f}")