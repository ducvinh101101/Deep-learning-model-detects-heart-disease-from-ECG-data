import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.api.callbacks import Callback
from keras.api.models import Model
from keras.api.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Concatenate, BatchNormalization, Layer, LSTM
from keras.src.layers import Reshape, LayerNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from keras.api.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.api.preprocessing.sequence import pad_sequences
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.optimizers import Adam
from keras.api.regularizers import l2

# Custom MLPBlock converted to a Keras Layer
class MLPBlock(Layer):
    def __init__(self, embedding_dim, num_tokens, token_mixer_dim, channel_mixer_dim, **kwargs):
        super(MLPBlock, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.token_mixer_dim = token_mixer_dim
        self.channel_mixer_dim = channel_mixer_dim

        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.token_mixer_fc1 = Dense(token_mixer_dim, activation='gelu')
        self.token_mixer_fc2 = Dense(num_tokens)

        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.channel_mixer_fc1 = Dense(channel_mixer_dim, activation='gelu')
        self.channel_mixer_fc2 = Dense(embedding_dim)

    def call(self, x):
        # Token Mixing
        x = self.layer_norm1(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.token_mixer_fc1(x)
        x = self.token_mixer_fc2(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        # Channel Mixing
        x = self.layer_norm2(x)
        x = self.channel_mixer_fc1(x)
        x = self.channel_mixer_fc2(x)
        return x

# Custom Feature-wise Attention Layer
class FeatureWiseAttention(Layer):
    def __init__(self, input_dim, **kwargs):
        super(FeatureWiseAttention, self).__init__(**kwargs)
        self.attention_layer = Dense(input_dim, activation='softmax')

    def call(self, inputs):
        attention_weights = self.attention_layer(inputs)
        return inputs * attention_weights

class FFTLayer(Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Perform FFT and use the magnitude
        fft = tf.signal.fft(tf.cast(inputs, tf.complex64))
        magnitude = tf.math.abs(fft)
        return tf.cast(magnitude, tf.float32)

# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


# Custom Scaled Dot-Product Attention Layer in Keras
class ScaledDotProductAttention(Layer):
    def __init__(self, temperature, attn_dropout=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.temperature = temperature
        self.dropout = Dropout(attn_dropout)

    def call(self, q, k, v, mask=None):
        attn = tf.matmul(q, k, transpose_b=True) / self.temperature

        if mask is not None:
            mask = tf.cast(mask, dtype=attn.dtype)
            attn = tf.where(mask == 0, -1e9, attn)

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        output = tf.matmul(attn, v)

        return output, attn

# Custom Abs Layer
class AbsLayer(Layer):
    def __init__(self, **kwargs):
        super(AbsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.abs(inputs)

# Custom layer to add dimension
class ExpandDimsLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

# Path to data
diagnostics_file = "Data_ECG/Diagnostics.csv"
ecg_data_folder = "Data_ECG/ECGData/ECGData"


# 1. Load Diagnostics.csv
diagnostics_data = pd.read_csv(diagnostics_file, encoding='utf-8')
diagnostics_data = diagnostics_data[
    ~diagnostics_data['Rhythm'].isin(['SAAWR', 'AVRT', 'AVNRT', 'AT', 'SA', 'AF', 'SVT'])].reset_index(drop=True)

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
diagnostics_data['Beat'] = diagnostics_data['Beat'].fillna('NONE')
diagnostics_data['Beat'] = diagnostics_data['Beat'].apply(lambda x: x.split())

# Mã hóa nhãn bằng MultiLabelBinarizer
mlb = MultiLabelBinarizer()
beat_encoded = mlb.fit_transform(diagnostics_data['Beat'])

# Scale numerical features
scaler = StandardScaler()
metadata_scaled_numeric = scaler.fit_transform(metadata_encoded.select_dtypes(include=[np.number]))
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
X_ecg_train = np.nan_to_num(X_ecg_train, nan=0.0, posinf=1e10, neginf=-1e10)
X_metadata_train = np.nan_to_num(X_metadata_train, nan=0.0, posinf=1e10, neginf=-1e10)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e10, neginf=-1e10)
# Model building
input_ecg = Input(shape=(X_ecg_train.shape[1], X_ecg_train.shape[2]))

x_ecg = Conv1D(filters=32, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(input_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = Conv1D(filters=32, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = Conv1D(filters=256, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = Conv1D(filters=256, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)
x_fft = FFTLayer()(x_ecg)
x_ecg = Concatenate()([x_ecg, x_fft])
x_ecg = MaxPooling1D(pool_size=2)(x_ecg)
x_ecg = Dropout(0.2)(x_ecg)
x_ecg = BatchNormalization()(x_ecg)
x_ecg = LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x_ecg)
x_ecg = LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x_ecg)
# Prepare Q, K, V for Attention
q = Dense(64)(x_ecg)
k = Dense(64)(x_ecg)
v = Dense(64)(x_ecg)
# Apply Scaled Dot-Product Attention
attention_layer = ScaledDotProductAttention(temperature=64**0.5)
x_ecg, _ = attention_layer(q, k, v)
x_ecg = Flatten()(x_ecg)
# Continue with pipeline
x_ecg = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_ecg)

# # Apply Feature Wise Attention
# x_ecg = FeatureWiseAttention(input_dim=x_ecg.shape[-1])(x_ecg)
# x_ecg = Flatten()(x_ecg)


# # Apply MLPBlock
# x_ecg = ExpandDimsLayer(axis=1)(x_ecg)
# mlp_block = MLPBlock(embedding_dim=512, num_tokens=1, token_mixer_dim=256, channel_mixer_dim=512)
# x_ecg = mlp_block(x_ecg)
# x_ecg = Flatten()(x_ecg)

input_metadata = Input(shape=(X_metadata_train.shape[1],))
x_metadata = Reshape((X_metadata_train.shape[1], 1))(input_metadata)
x_metadata = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x_metadata)
x_metadata = MaxPooling1D(pool_size=2)(x_metadata)
x_metadata = Dropout(0.2)(x_metadata)
x_metadata = BatchNormalization()(x_metadata)
x_metadata = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x_metadata)
x_metadata = MaxPooling1D(pool_size=2)(x_metadata)
x_metadata = Dropout(0.2)(x_metadata)
x_metadata = BatchNormalization()(x_metadata)
x_metadata = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x_metadata)
x_metadata = Flatten()(x_metadata)

combined = Concatenate()([x_ecg, x_metadata])
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(combined)
x = Dropout(0.3)(x)
output = Dense(y_train.shape[1], activation='softmax')(x)

# Compile model
optimizer = Adam(learning_rate=1e-4)
model = Model(inputs=[input_ecg, input_metadata], outputs=output)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# Initialize variables for best model tracking
best_accuracy = 0.0
best_weights_path = "best_model_weights.weights.h5"

# Custom callback to track best model
class BestModelCheckpoint(Callback):
    def __init__(self, best_weights_path):
        super(BestModelCheckpoint, self).__init__()
        self.best_weights_path = best_weights_path
        self.best_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('val_accuracy')
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.model.save_weights(self.best_weights_path)

# Add the custom callback
best_model_checkpoint = BestModelCheckpoint(best_weights_path)

# Train the model
history = model.fit(
    [X_ecg_train, X_metadata_train], y_train,
    validation_data=([X_ecg_test, X_metadata_test], y_test),
    epochs=50, batch_size=64,
    callbacks=[best_model_checkpoint]
)

# Load the best weights after training
model.load_weights(best_weights_path)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_ecg_test, X_metadata_test], y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model and label encoder
model.save("ecg_combined_model.h5")
with open("label_encoder.pkl", "wb") as f:
    import pickle
    pickle.dump(label_encoder, f)

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion matrix
y_pred = model.predict([X_ecg_test, X_metadata_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
class_names = label_encoder.classes_
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names).plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()
