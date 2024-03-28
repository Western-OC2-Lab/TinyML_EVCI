import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


def plot_histogram(data, columnName, unit, title):
    # Load the data
    # data = pd.read_csv(dataName + '.csv')
    n_samples = len(data)

    # Define the font settings
    calibri_font = {'family': 'Calibri',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 20,
                    }
    # Plot the histogram of the data
    k = int(1 + 3.322 * math.log((data.shape[0]), 2))
    data[columnName].hist(bins=k, density=False, alpha=1, edgecolor='black')  # label='Histogram'
    plt.xlabel(f'{columnName} ({unit})', fontsize=50, labelpad=15, fontdict=calibri_font)
    plt.ylabel('Density', fontsize=50, labelpad=15, fontdict=calibri_font)
    plt.title(title, fontsize=50, pad=22, fontdict=calibri_font)

    # Calculate the mean and standard deviation of the data
    mu, std = data[columnName].mean(), data[columnName].std()

    # Plot the normal curve
    xmin, xmax = plt.xlim()
    bin_width = (xmax - xmin) / k
    x = np.linspace(xmin - 3 * std, xmax + 3 * std, 1000)
    p = stats.norm.pdf(x, mu, std) * n_samples * bin_width
    plt.plot(x, p, linewidth=3, color='#993d4d')  # label='Normal Distribution'

    minumum = data.min().min()
    maximum = data.max().max()

    # Create custom legend labels
    legend_labels = [
        f'Min:{minumum:.2f}',
        f'Max: {maximum:.2f}',
        f'Mean: {mu:.2f}',
        f'Std: {std:.2f}',
        f'Number of Samples: {n_samples}',
    ]

    # Create a custom legend using dummy plot objects with no lines or markers
    for label in legend_labels:
        plt.plot([], [], ' ', label=label)

    # Display the legend with specified font size
    plt.legend(fontsize=35)

    plt.xticks(fontsize=36, fontname="Calibri")
    plt.yticks(fontsize=36, fontname="Calibri")

    plt.rcParams['font.family'] = 'Calibri'
    plt.show()

#1
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\ML_MLP_PerSampleMemory_KB.csv'
    var_name = 'ML_MLP_PerSampleMemory_KB' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#2
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\ML_MLP_PerSamplePredictionTime_ms.csv'
    var_name = 'ML_MLP_PerSamplePredictionTime_ms' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#3
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\ML_RF_PerSampleMemory_KB.csv'
    var_name = 'ML_RF_PerSampleMemory_KB' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#4
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\ML_RF_PerSamplePredictionTime_ms.csv'
    var_name = 'ML_RF_PerSamplePredictionTime_ms' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#5
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\TinyML_MLP_PerSampleMemory_Bytes.csv'
    var_name = 'TinyML_MLP_PerSampleMemory_Bytes' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#6
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\TinyML_MLP_PerSamplePredictionTime_us.csv'
    var_name = 'TinyML_MLP_PerSamplePredictionTime_us' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#7
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\TinyML_RF_PerSampleMemory_KB.csv'
    var_name = 'TinyML_RF_PerSampleMemory_KB' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

#8
for i in range(1, 6):
    file_path = r'C:\\Users\\FDEHROUY\\Desktop\\Run\\Fold' + str(i) + r'\\TinyML_RF_PerSamplePredictionTime_ms.csv'
    var_name = 'TinyML_RF_PerSamplePredictionTime_ms' + str(i)
    globals()[var_name] = pd.read_csv(file_path)

ML_MLP_PerSampleMemory_KB = pd.concat([ML_MLP_PerSampleMemory_KB1, ML_MLP_PerSampleMemory_KB2, ML_MLP_PerSampleMemory_KB3, ML_MLP_PerSampleMemory_KB4, ML_MLP_PerSampleMemory_KB5])
ML_MLP_PerSamplePredictionTime_ms = pd.concat([ML_MLP_PerSamplePredictionTime_ms1, ML_MLP_PerSamplePredictionTime_ms2, ML_MLP_PerSamplePredictionTime_ms3, ML_MLP_PerSamplePredictionTime_ms4, ML_MLP_PerSamplePredictionTime_ms5])
ML_RF_PerSampleMemory_KB = pd.concat([ML_RF_PerSampleMemory_KB1, ML_RF_PerSampleMemory_KB2, ML_RF_PerSampleMemory_KB3, ML_RF_PerSampleMemory_KB4, ML_RF_PerSampleMemory_KB5])
ML_RF_PerSamplePredictionTime_ms = pd.concat([ML_RF_PerSamplePredictionTime_ms1, ML_RF_PerSamplePredictionTime_ms2, ML_RF_PerSamplePredictionTime_ms3, ML_RF_PerSamplePredictionTime_ms4, ML_RF_PerSamplePredictionTime_ms5])
TinyML_MLP_PerSampleMemory_Bytes = pd.concat([TinyML_MLP_PerSampleMemory_Bytes1, TinyML_MLP_PerSampleMemory_Bytes2, TinyML_MLP_PerSampleMemory_Bytes3, TinyML_MLP_PerSampleMemory_Bytes4, TinyML_MLP_PerSampleMemory_Bytes5])
TinyML_MLP_PerSamplePredictionTime_us = pd.concat([TinyML_MLP_PerSamplePredictionTime_us1, TinyML_MLP_PerSamplePredictionTime_us2, TinyML_MLP_PerSamplePredictionTime_us3, TinyML_MLP_PerSamplePredictionTime_us4, TinyML_MLP_PerSamplePredictionTime_us5])
TinyML_RF_PerSampleMemory_KB = pd.concat([TinyML_RF_PerSampleMemory_KB1, TinyML_RF_PerSampleMemory_KB2, TinyML_RF_PerSampleMemory_KB3, TinyML_RF_PerSampleMemory_KB4, TinyML_RF_PerSampleMemory_KB5])
TinyML_RF_PerSamplePredictionTime_ms = pd.concat([TinyML_RF_PerSamplePredictionTime_ms1, TinyML_RF_PerSamplePredictionTime_ms2, TinyML_RF_PerSamplePredictionTime_ms3, TinyML_RF_PerSamplePredictionTime_ms4, TinyML_RF_PerSamplePredictionTime_ms5])



plot_histogram(ML_MLP_PerSampleMemory_KB, 'Memory' , 'KB', 'ML_MLP Memory Consumption')
plot_histogram(ML_RF_PerSampleMemory_KB, 'Memory','KB', 'ML_RF Memory Consumption')
plot_histogram(TinyML_MLP_PerSampleMemory_Bytes, 'Memory', 'Bytes', 'TinyML_MLP Memory Consumption')
plot_histogram(TinyML_RF_PerSampleMemory_KB, 'Memory', 'KB', 'TinyML_RF Memory Consumption')

plot_histogram(ML_MLP_PerSamplePredictionTime_ms, 'Time', 'ms', 'ML_MLP Inference Time')
plot_histogram(ML_RF_PerSamplePredictionTime_ms, 'Time','ms', 'ML_RF Inference Time')
plot_histogram(TinyML_MLP_PerSamplePredictionTime_us, 'Time','\u03BC' + 's', 'TinyML_MLP Inference Time')
plot_histogram(TinyML_RF_PerSamplePredictionTime_ms, 'Time', 'ms', 'TinyML_RF Inference Time')


negative_values = TinyML_MLP_PerSampleMemory_Bytes[TinyML_MLP_PerSampleMemory_Bytes.iloc[:,0]<0]

if not negative_values.empty:
    print("The CSV contains negative values:")
    print(negative_values)
else:
    print("The CSV does not contain any negative values.")
