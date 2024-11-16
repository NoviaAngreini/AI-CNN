import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
 
# Memuat data CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
 
# Normalisasi data gambar
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
 
# Mengonversi label ke bentuk kategorikal
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
 
# Sekarang, 'train_images' dan 'train_labels' siap digunakan untuk pelatihan model.
 
 
# Membangun model CNN
model = tf.keras.Sequential([
    # Layer konvolusi pertama
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
 
    # Layer konvolusi kedua
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
 
    # Layer konvolusi ketiga
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
 
    # Flattening output untuk menginputkannya ke dalam Dense layer
    tf.keras.layers.Flatten(),
 
    # Dense layer dengan Dropout
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
 
    # Layer output
    tf.keras.layers.Dense(10, activation='softmax') # dikarenakan ada 10 kelas
])
 
# Menampilkan ringkasan model
model.summary()
 
# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 
# Pelatihan model
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
 
 
# Evaluasi model pada data tes
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

from google.colab import files
from keras.models import load_model
from PIL import Image
import numpy as np
 
# Fungsi untuk memuat dan memproses gambar
def load_and_prepare_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))  # Sesuaikan dengan dimensi yang model Anda harapkan
    img = np.array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension
    return img
 
# Nama-nama kelas untuk dataset CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
# Muat model
#model = load_model('path_to_your_model.h5')  # Ganti dengan path ke model yang disimpan
 
# Mengunggah file gambar
uploaded = files.upload()
 
# Memproses gambar dan membuat prediksi
for filename in uploaded.keys():
    img = load_and_prepare_image(filename)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Dapatkan indeks kelas
    predicted_class_name = class_names[predicted_class_index]  # Dapatkan nama kelas
    print(f'File: {filename}, Predicted Class Index: {predicted_class_index}, Predicted Class Name: {predicted_class_name}')
 
