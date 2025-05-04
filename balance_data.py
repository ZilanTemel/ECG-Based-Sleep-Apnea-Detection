import numpy as np
from sklearn.utils import resample, shuffle

# load X and y
X = np.load("X.npy")
y = np.load("y.npy")

# separate apnea = 0 and apnea = 1
X_0 = X[y == 0]
X_1 = X[y == 1]

# oversample class 1 to match class 0 size
X_1_resampled, y_1_resampled = resample(X_1, [1]*len(X_1), replace=True, n_samples=len(X_0), random_state=42)

# combine and shuffle the data
X_balanced = np.concatenate([X_0, X_1_resampled])
y_balanced = np.concatenate([[0]*len(X_0), y_1_resampled])
X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

# save the new balanced data
np.save("X_balanced.npy", X_balanced)
np.save("y_balanced.npy", y_balanced)

print(" Balanced data saved.")
