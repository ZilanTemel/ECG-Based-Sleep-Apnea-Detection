import os
import wfdb
import matplotlib.pyplot as plt

# path to the dataset folder
data_path = "C:/Yazılım/Machine learning project/apnea-ecg/"

# list all files in the folder
file_list = os.listdir(data_path)
print("Files found:", file_list)

# get file IDs from .hea files
file_ids = sorted(list(set([f.split('.')[0] for f in file_list if f.endswith('.hea')])))

if not file_ids:
    raise Exception("No .hea files found! Check the path and folder content.")

sample_id = file_ids[0]
print(f"Processing file: {sample_id}")

# read signal and annotations
record = wfdb.rdrecord(os.path.join(data_path, sample_id))
annotation = wfdb.rdann(os.path.join(data_path, sample_id), 'qrs')

# plot signal and annotations
signal = record.p_signal[:, 0]
fs = record.fs

plt.figure(figsize=(15, 4))
plt.plot(signal, label='ECG Signal')
plt.scatter(annotation.sample, [signal[i] for i in annotation.sample], color='red', label='QRS Annotations', s=10)
plt.title(f"{sample_id} - ECG Signal with QRS Annotations")
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
