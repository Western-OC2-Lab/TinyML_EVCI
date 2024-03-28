import pandas as pd
import numpy as np

methods = ['ML_MLP_outputs', 'ML_RF_outputs', 'TinyML_MLP_outputs', 'TinyML_RF_outputs']

for name in methods:
    for i in range(1, 6):
        file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\' + str(name) + r'.csv'
        var_name = name + str(i)
        globals()[var_name] = pd.read_csv(file_path)

# ML_MLP_outputs
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'training_time_s',
            'training_memory_consumption_MB', 'Average_prediction_time_ms',
            'Average_prediction_memory_consumption_KB', 'size_KB']
data = []
for metric in metrics:
    list_ = 'ML_MLP_' + str(metric)
    list_ = []
    for i in range(1,6):
        current_dataframe = globals()['ML_MLP_outputs' + str(i)]
        value = current_dataframe[current_dataframe['Metrics'] == metric]['Values'].values[0]
        list_.append(value)

    mean_value = sum(list_)/len(list_)
    data.append([metric, mean_value])

data = pd.DataFrame(data, columns = ['Metric', 'Calculated_Value'])
data.to_csv('ML_MLP_Average.csv', index = False)

# ML_RF_outputs
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score',
'training_time_ms', 'training_memory_KB', 'Average_prediction_time_ms',
 'Average_prediction_memory_consumption_KB', 'size_KB']

data = []
for metric in metrics:
    list_ = 'ML_RF_' + str(metric)
    list_ = []
    for i in range(1,6):
        current_dataframe = globals()['ML_RF_outputs' + str(i)]
        value = current_dataframe[current_dataframe['Metrics'] == metric]['Values'].values[0]
        list_.append(value)

    mean_value = sum(list_)/len(list_)
    data.append([metric, mean_value])

data = pd.DataFrame(data, columns = ['Metric', 'Calculated_Value'])
data.to_csv('ML_RF_Average.csv', index = False)

# TinyML_MLP_outputs
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score',
           'conversion_time_s', 'Average_prediction_time_us',
           'Average_prediction_memory_consumption_Bytes', 'size_KB']

data = []
for metric in metrics:
    list_ = 'TinyML_MLP_' + str(metric)
    list_ = []
    for i in range(1,6):
        current_dataframe = globals()['TinyML_MLP_outputs' + str(i)]
        value = current_dataframe[current_dataframe['Metrics'] == metric]['Values'].values[0]
        list_.append(value)

    mean_value = sum(list_)/len(list_)
    data.append([metric, mean_value])

data = pd.DataFrame(data, columns = ['Metric', 'Calculated_Value'])
data.to_csv('TinyML_MLP_Average.csv', index = False)

# TinyML_RF
metrics = ['Accuracy', 'Precision', 'Recall','F1_Score',
           'training_time_ms', 'training_memory_KB', 'Average_prediction_time_ms',
           'Average_prediction_memory_consumption_KB', 'size_KB']

data = []
for metric in metrics:
    list_ = 'TinyML_RF_' + str(metric)
    list_ = []
    for i in range(1,6):
        current_dataframe = globals()['TinyML_RF_outputs' + str(i)]
        value = current_dataframe[current_dataframe['Metrics'] == metric]['Values'].values[0]
        list_.append(value)

    mean_value = sum(list_)/len(list_)
    data.append([metric, mean_value])

data = pd.DataFrame(data, columns = ['Metric', 'Calculated_Value'])
data.to_csv('TinyML_RF_Average.csv', index = False)