import os
import pandas as pd
import numpy as np
from keras.api.models import Model
from keras.api.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, Input, Concatenate
from keras.src.layers import Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from keras.api.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.api.preprocessing.sequence import pad_sequences
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.optimizers import Adam
from keras.api.regularizers import l2

# Path to data
diagnostics_file = "Data_ECG/Diagnostics.csv"
ecg_data_folder = "Data_ECG/ECGData/ECGData"

# 1. Load Diagnostics.csv
diagnostics_data = pd.read_csv(diagnostics_file, encoding='utf-8')
diagnostics_data = diagnostics_data[~diagnostics_data['Rhythm'].isin(['SAAWR', 'AVRT', 'AVNRT', 'AT', 'SA', 'AF', 'SVT'])].reset_index(drop=True)

# Giảm số lượng SB xuống 1800 cái đầu
diagnostics_data_sb = diagnostics_data[diagnostics_data['Rhythm'] == 'SB']
diagnostics_data_non_sb = diagnostics_data[diagnostics_data['Rhythm'] != 'SB']
if len(diagnostics_data_sb) > 1800:
    diagnostics_data_sb = diagnostics_data_sb.iloc[:1800]
diagnostics_data = pd.concat([diagnostics_data_sb, diagnostics_data_non_sb]).reset_index(drop=True)

# Extract features from Diagnostics.csv
metadata_features = diagnostics_data.drop(columns=['FileName', 'Rhythm'])
label_column = diagnostics_data['Rhythm']

metadata_encoded = pd.get_dummies(metadata_features, columns=['Gender'], drop_first=True)

# Process Beat column
diagnostics_data['Beat'] = diagnostics_data['Beat'].fillna('NONE')  # Thay thế giá trị NaN bằng 'NONE'
diagnostics_data['Beat'] = diagnostics_data['Beat'].apply(lambda x: x.split())  # Tách các nhãn thành danh sách

# Mã hóa nhãn bằng MultiLabelBinarizer
mlb = MultiLabelBinarizer()
beat_encoded = mlb.fit_transform(diagnostics_data['Beat'])
beat_columns = mlb.classes_

# Filter numeric columns for scaling
numeric_columns = metadata_encoded.select_dtypes(include=[np.number])

# Scale numerical features
scaler = StandardScaler()
metadata_scaled_numeric = scaler.fit_transform(numeric_columns)

# Combine metadata and beat data
metadata_scaled = np.hstack([metadata_scaled_numeric, beat_encoded])

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(label_column)
labels_one_hot = to_categorical(labels_encoded)

# 2. Load ECG data from CSV files
def load_ecg_data(file_name):
    file_path = os.path.join(ecg_data_folder, file_name)
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path).iloc[1:].to_numpy()
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            return None
    else:
        print(f"File {file_name} not found!")
        return None

ecgs = []
metadata_list = []
labels = []

for idx, row in diagnostics_data.iterrows():
    ecg_file = row['FileName'] + ".csv"
    print(f"Loading {ecg_file}...")
    ecg_signal = load_ecg_data(ecg_file)
    if ecg_signal is None or ecg_signal.size == 0:
        print(f"Skipping {ecg_file} due to empty or invalid data.")
        continue
    ecgs.append(ecg_signal)
    metadata_list.append(metadata_scaled[idx])
    labels.append(row['Rhythm'])
print("Loading complete!")

# 3. Pad and preprocess ECG signals
max_length = max(len(signal) for signal in ecgs)
ecgs_padded = pad_sequences(ecgs, maxlen=max_length, padding='post', dtype='float32')
print(2)
X_ecg = ecgs_padded
X_metadata = np.array(metadata_list)
y = labels_one_hot

min_samples = min(len(X_ecg), len(X_metadata), len(y))
X_ecg = X_ecg[:min_samples]
X_metadata = X_metadata[:min_samples]
y = y[:min_samples]
print(3)

# Train-test split
X_ecg_train, X_ecg_test, X_metadata_train, X_metadata_test, y_train, y_test = train_test_split(
    X_ecg, X_metadata, y, test_size=0.2, random_state=42
)
print(4)

# 4. Build model
# Chuẩn hóa dữ liệu để tránh NaN/Inf
X_ecg_train = np.nan_to_num(X_ecg_train, nan=0.0, posinf=1e10, neginf=-1e10)
X_metadata_train = np.nan_to_num(X_metadata_train, nan=0.0, posinf=1e10, neginf=-1e10)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e10, neginf=-1e10)

input_ecg = Input(shape=(X_ecg_train.shape[1], X_ecg_train.shape[2]))

x_ecg = Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.001))(input_ecg)
x_ecg = Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_ecg = Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_ecg = Conv1D(filters=128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_ecg = Flatten()(x_ecg)

input_metadata = Input(shape=(X_metadata_train.shape[1],))
x_metadata = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(input_metadata)
x_metadata = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x_metadata)

combined = Concatenate()([x_ecg, x_metadata])
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
x = Dropout(0.3)(x)
output = Dense(y_train.shape[1], activation='softmax')(x)

# Compile model
optimizer = Adam(learning_rate=1e-4)
model = Model(inputs=[input_ecg, input_metadata], outputs=output)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# Train the model
history = model.fit(
    [X_ecg_train, X_metadata_train], y_train,
    validation_data=([X_ecg_test, X_metadata_test], y_test),
    epochs=50, batch_size=64,
    callbacks=[early_stopping, lr_reduction]
)

# 6. Evaluate the model
test_loss, test_accuracy = model.evaluate([X_ecg_test, X_metadata_test], y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 7. Save the model and label encoder
model.save("ecg_combined_model.h5")
with open("label_encoder.pkl", "wb") as f:
    import pickle
    pickle.dump(label_encoder, f)

# 8. Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 9. Confusion matrix
y_pred = model.predict([X_ecg_test, X_metadata_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
class_names = label_encoder.classes_
# Lấy nhãn thực tế từ dữ liệu
unique_labels = np.unique(np.concatenate((y_true_classes, y_pred_classes)))
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
# Tạo ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names).plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()
