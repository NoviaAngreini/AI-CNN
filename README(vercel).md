# AI VERCEL CNN

Berikut adalah deskripsi CNN Vercel

---

# 1. Memuat dan Menyiapkan Data CIFAR-10
- Dataset CIFAR-10:  Dataset ini berisi 60.000 gambar berwarna (32x32 piksel) dalam 10 kelas, seperti pesawat, mobil, burung, dsb.
  
```python
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```

- Normalisasi Gambar:   Gambar diubah ke nilai float dalam rentang [0, 1] agar lebih stabil selama pelatihan.

```python
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
```

- Label Kategorikal:  Label dikonversi ke bentuk *one-hot encoding* menggunakan `to_categorical` karena output model memiliki 10 kelas.

```python
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
```

# 2. Membangun Model CNN
- Model CNN terdiri dari beberapa layer:
  1. Layer Konvolusi dan Pooling:     Ekstraksi fitur dari gambar melalui filter convolusi dan pengurangan dimensi dengan pooling.
  2. Flatten:     Mengubah hasil feature map menjadi vektor 1D.
  3. Dense Layer:     Kombinasi non-linearitas untuk klasifikasi, termasuk penggunaan Dropout untuk mengurangi overfitting.
  4. Output Layer:     Menggunakan aktivasi `softmax` untuk menghasilkan probabilitas dari 10 kelas.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

# 3. Kompilasi Model
- Optimizer: Adam (efisien dan adaptif).
- Loss: Categorical Crossentropy (sesuai untuk masalah multi-klasifikasi).
- Metrik: Akurasi.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

# 4. Pelatihan Model
- `fit` digunakan untuk melatih model:
  - Epoch: 10.
  - Batch size: 64.
  - Validasi dilakukan dengan data tes.

```python
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```

# 5. Evaluasi Model
- `evaluate` mengukur performa model pada data tes.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

# 6. Prediksi pada Gambar Baru
a. Memproses Gambar
- Fungsi untuk memuat gambar, mengubah ukurannya menjadi 32x32 piksel, melakukan normalisasi, dan menambah dimensi batch.

```python
def load_and_prepare_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
```

b. Prediksi Kelas
- Model memprediksi kelas gambar yang diunggah.
- Nama kelas diperoleh menggunakan indeks prediksi.

```python
uploaded = files.upload()  # Mengunggah file gambar
for filename in uploaded.keys():
    img = load_and_prepare_image(filename)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    print(f'File: {filename}, Predicted Class Index: {predicted_class_index}, Predicted Class Name: {predicted_class_name}')
```

# Output
- Ringkasan Model: Informasi arsitektur model.
- Akurasi Tes: Evaluasi performa pada data tes CIFAR-10.
- Prediksi Gambar: Menampilkan prediksi kelas gambar yang diunggah beserta nama kelasnya.
