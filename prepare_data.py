import wfdb
import numpy as np
import os

data_path = "apnea-ecg"

prefixes = ['a', 'b', 'c', 'x']
ranges = {'a': 20, 'b': 5, 'c': 10, 'x': 35}

X = []
y = []

for prefix in prefixes:
    for i in range(1, ranges[prefix] + 1):
        rec_id = f"{prefix}{str(i).zfill(2)}"
        try:
            record = wfdb.rdrecord(os.path.join(data_path, rec_id))
            signal = record.p_signal[:, 0]

            ann = wfdb.rdann(os.path.join(data_path, rec_id + "r"), "apn")
            labels = ann.symbol

            fs = int(record.fs)
            minute_samples = fs * 60

            for j in range(len(labels)):
                start = j * minute_samples
                end = start + minute_samples
                if end <= len(signal):
                    X.append(signal[start:end])
                    y.append(1 if labels[j] == 'A' else 0)

            print(f"{rec_id} processed.")
        except Exception as e:
            print(f"{rec_id} skipped. Error: {e}")

X = np.array(X).reshape((-1, minute_samples, 1))
y = np.array(y)

# SAVE DATA (final step)
np.save("X.npy", X)
np.save("y.npy", y)
print(" X and y successfully saved as .npy files.")

print("All data processed.")
print("Total number of samples:", len(X))
print("Label distribution:", np.unique(y, return_counts=True))
