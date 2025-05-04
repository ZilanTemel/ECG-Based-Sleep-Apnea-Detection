# Supervised Learning for Sleep Apnea Detection Based on ECG

This project uses an LSTM-based deep learning model to detect sleep apnea using ECG (Electrocardiogram) signals. The goal is to classify whether individuals experience apnea during sleep.

## 1. Dataset Description

## Dataset

# You can download the dataset used in this project from the following link: [Kaggle - Apnea-ECG Database](https://www.kaggle.com/datasets/ecerulm/apneaecg)

# You need to place the **X.npy** and **y.npy** files in the project folder after downloading the dataset. Once the files are placed correctly in the project directory, you can run the code.

- **Source:** Apnea-ECG Dataset (from Kaggle)
- **Content:** Multichannel ECG signals in .dat, .hea, .apn files
- **Size:** Over 70 patient records
- **Feature:** Each record contains multiple minutes of signal data and annotations.

## 2. Preprocessing

- **Signal Reading:** ECG signals were read using the `wfdb` library.
- **Annotations:** QRS annotations were marked, and samples were extracted.
- **Data Format:** Each window was saved as `X.npy`, and labels were saved as `y.npy`.
- **Reshaping:** Data was reshaped without scaling to make it compatible with LSTM: `(number_of_samples, time_step, 1)`.

## 3. Model Training and Testing

- **Model Type:** Bidirectional LSTM + Dropout + Batch Normalization
- **Data Split:** 80% Training - 20% Testing
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 10
- **Class Balance:** Managed using `class_weight`.

### Model Architecture

```python
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(64)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

## 4. Model Evaluation

The model was evaluated on the test data using `classification_report`:

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0 (No Apnea)  | 0.76      | 0.87   | 0.81     | 459     |
| 1 (Apnea)     | 0.78      | 0.63   | 0.70     | 332     |

- **Accuracy:** 0.77
- **Macro Avg:** 0.77 / 0.75 / 0.75
- **Weighted Avg:** 0.77 / 0.77 / 0.77
