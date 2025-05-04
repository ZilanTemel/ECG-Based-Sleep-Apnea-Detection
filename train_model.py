# 1️ Necessary libraries were imported
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

# 2️ The data was loaded
X = np.load("X.npy")
y = np.load("y.npy")

# 3️ The dataset was split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️ Class weights were calculated to handle class imbalance
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print("Class weights:", cw_dict)

# 5️ The LSTM-based model was built
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(64)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 6️ The model was compiled
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7️ The model summary was printed
model.summary()

# 8️ The model was trained with the training data
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=cw_dict
)

# 9️ The model was tested and evaluation results were printed
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nTest Results:")
print(classification_report(y_test, y_pred))
