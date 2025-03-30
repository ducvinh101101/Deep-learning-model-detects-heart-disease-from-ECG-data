import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pandas as pd
from keras.api.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report
from keras.api.optimizers import Adam

Model_name = 'MinimizeVGG'

# Tạo đường dẫn train test
Train_dir = "/content/drive/MyDrive/BrainTumor4Class_Rz128/Training"
Test_dir = "/content/drive/MyDrive/BrainTumor4Class_Rz128/Testing"

# Tạo các biến thể của ảnh ( Làm giàu dữ liệu ảnh)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load ảnh đầu vào (thay bằng đường dẫn ảnh của bạn)
img_path = Train_dir + '/notumor/Tr-no_1022.jpg'  # Thay bằng đường dẫn ảnh của bạn
img = load_img(img_path, target_size=(128, 128))  # Load ảnh và resize
img_array = img_to_array(img)  # Chuyển thành array NumPy
img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

# Tạo một lô ảnh biến đổi từ ảnh đầu vào
augmented_images = train_datagen.flow(img_array, batch_size=1)

# Hiển thị 6 ảnh đã được biến đổi
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    batch = next(augmented_images)  # Lấy một ảnh biến đổi
    img_augmented = batch[0]  # Lấy ảnh đầu tiên từ batch
    plt.imshow(img_augmented)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Chuẩn hóa cho tập dữ liệu test
test_datagen = ImageDataGenerator(rescale=1./255)

# Load ảnh từ thư mục và áp dụng xử lý dữ liệu
train_generator = train_datagen.flow_from_directory(
    Train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    Test_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Xây mô hình VGG

model = Sequential()

# Block 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Block 4
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Classification Head
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Compile mô hình
optimizer = Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Đưa test vào theo dõi quá trình
test_losses = []
test_accuracies = []
class TestMetricsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = model.evaluate(test_generator, verbose=0)
        if len(test_accuracies) > 0 and test_acc >= max(test_accuracies):
            model.save(f'/content/drive/MyDrive/BrainTumor4Class_Trains/{Model_name}/best.h5')
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f" - Test loss = {test_loss} - Test accuracy = {test_acc}")
        model.save(f'/content/drive/MyDrive/BrainTumor4Class_Trains/{Model_name}/last.h5')

# Huấn luyện mô hình mà không có tập validation
H = model.fit(
    train_generator,
    epochs=100,
    verbose=1,
    callbacks=[TestMetricsCallback()]
)

# Đánh giá mô hình qua test
score = model.evaluate(test_generator, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Vẽ biểu đồ chính xác và độ mất mát trong quá trình huấn luyện
fig = plt.figure()
numOfEpoch = 100
# plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
# plt.plot(np.arange(0, numOfEpoch), test_losses, label='test loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='training accuracy')
# plt.plot(np.arange(0, numOfEpoch), test_accuracies, label='test accuracy')
plt.title('Accuracy - Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'/content/drive/MyDrive/BrainTumor4Class_Trains/{Model_name}/train_accu.png')
plt.show()

# plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='training accuracy')
plt.plot(np.arange(0, numOfEpoch), test_accuracies, label='test accuracy')
plt.title('Accuracy - Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'/content/drive/MyDrive/BrainTumor4Class_Trains/{Model_name}/test_accu.png')
plt.show()

# Ma trận nhầm lẫn và báo cáo
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Tạo ma trận nhầm lẫn (Confusion matrix)
cm = confusion_matrix(y_true, y_pred)
labels = ['glioma', 'meningioma', 'pituitary', 'notumor']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'/content/drive/MyDrive/BrainTumor4Class_Trains/{Model_name}/confusion_matrix.png')
plt.show()

# Tạo báo cáo phân loại dưới dạng DataFrame
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Vẽ bảng từ DataFrame
fig, ax = plt.subplots(figsize=(10, 6))  # Kích thước hình ảnh
ax.axis('tight')
ax.axis('off')
table = plt.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center', cellLoc='center')

# Lưu bảng thành ảnh
plt.savefig(f'/content/drive/MyDrive/BrainTumor4Class_Trains/{Model_name}/classification_report.png', dpi=300, bbox_inches='tight')

# In báo cáo phân loại (precision, recall, f1-score)
print(classification_report(y_true, y_pred, target_names=labels))

print(H.history['loss'])
print(H.history['accuracy'])
print(test_losses)
print(test_accuracies)