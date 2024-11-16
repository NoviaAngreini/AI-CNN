# AI-CNN
Berikut adalah penjelasan dari CNN APLIKASI link ke 2

---

# 1. Mengimpor Library yang Diperlukan
- `tensorflow.keras.models.Sequential`:  Digunakan untuk membuat model jaringan saraf tiruan secara sekuensial, di mana layer ditambahkan satu per satu.
- `tensorflow.keras.layers.Conv2D`:  Layer convolusi untuk mendeteksi fitur dalam gambar.
- `tensorflow.keras.layers.MaxPooling2D`:  Layer pooling untuk mengurangi dimensi spasial fitur peta (spatial feature maps) sekaligus mengurangi risiko overfitting.
- `tensorflow.keras.layers.Flatten`:  Digunakan untuk mengubah data 2D menjadi vektor 1D untuk input ke layer fully connected.
- `tensorflow.keras.layers.Dense`: Layer fully connected untuk menghasilkan output prediksi.
- `tensorflow.keras.preprocessing.image.ImageDataGenerator`:  Digunakan untuk preprocessing data gambar, termasuk augmentasi data (rescaling, rotasi, flipping, dsb).

---

# 2. Inisialisasi Model CNN
`MesinKlasifikasi = Sequential()`:  Model sekuensial diinisialisasi untuk membangun CNN langkah demi langkah.

# 3. Langkah-Langkah Arsitektur CNN

a. Langkah 1 - Convolution

```python
MesinKlasifikasi.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape = (128, 128, 3), activation = 'relu'))
```
- `Conv2D` menambahkan layer convolusi:
    - `filters=32`: Menggunakan 32 filter.
    - `kernel_size=(3, 3)`: Ukuran filter adalah 3x3.
    - `input_shape=(128, 128, 3)`: Input gambar memiliki dimensi 128x128 dengan 3 channel warna (RGB).
    - `activation='relu'`: Fungsi aktivasi ReLU digunakan untuk mempercepat pembelajaran dengan menyingkirkan nilai negatif.

b. Langkah 2 - Pooling

```python
MesinKlasifikasi.add(MaxPooling2D(pool_size = (2, 2)))
```
- `MaxPooling2D` menambahkan layer pooling untuk mengurangi dimensi spasial dari fitur:
- `pool_size=(2, 2)`: Ukuran filter pooling adalah 2x2, mengambil nilai maksimum dari setiap area 2x2.

c. Menambah Layer Convolutional

```python
MesinKlasifikasi.add(Conv2D(32, (3, 3), activation = 'relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size = (2, 2)))
```
- Menambahkan layer convolusi dan pooling tambahan untuk menangkap fitur yang lebih kompleks.

d. Langkah 3 - Flattening
```python
MesinKlasifikasi.add(Flatten())
```
- `Flatten`: mengubah fitur 2D yang dihasilkan oleh pooling menjadi vektor 1D agar dapat diproses oleh layer fully connected.

e. Langkah 4 - Full Connection

```python
MesinKlasifikasi.add(Dense(units = 128, activation = 'relu'))
MesinKlasifikasi.add(Dense(units = 1, activation = 'sigmoid'))
```
- `Dense` menambahkan layer fully connected:
    - `units=128`: Layer tersembunyi dengan 128 neuron.
    - `activation='relu'`: Aktivasi ReLU.
    - Output layer:
        - `units=1`: Hanya 1 neuron output untuk klasifikasi biner.
        - `activation='sigmoid'`: Aktivasi sigmoid menghasilkan probabilitas antara 0 dan 1.


# 4. Kompilasi Model
```python
MesinKlasifikasi.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```
- `optimizer='adam'`: Algoritma optimasi Adam digunakan untuk memperbarui bobot model.
- `loss='binary_crossentropy'`: Fungsi loss digunakan untuk klasifikasi biner.
- `metrics=['accuracy']`: Metrik yang diukur adalah akurasi.

---

# 5. Persiapan Data
a. Augmentasi dan Normalisasi Data

```python
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
```
- Training data:
  -`rescale=1./255`: Normalisasi pixel ke rentang [0, 1].
  - Augmentasi:
    - `shear_range=0.2`: Mengubah sudut gambar.
    - `zoom_range=0.2`: Melakukan zoom secara acak.
    - `horizontal_flip=True`: Membalik gambar secara horizontal.
- Testing data:
  - Hanya dinormalisasi tanpa augmentasi.

b. Load Data dari Direktori\
```python
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')
```

- `flow_from_directory`: Memuat data gambar dari direktori dengan struktur folder.
  - `target_size=(128, 128)`: Gambar diubah ukurannya menjadi 128x128.
  - `batch_size=32`: Memproses gambar dalam batch 32.
  - `class_mode='binary'`: Label data dalam format biner (0 atau 1).

---

# 6. Melatih Model

```python
MesinKlasifikasi.fit_generator(training_set,
                               steps_per_epoch = 8000/32,
                               epochs = 50,
                               validation_data = test_set,
                               validation_steps = 2000/32)
```
- `fit_generator`:
  - `training_set`: Dataset pelatihan.
  - `steps_per_epoch=8000/32`: Model akan memperbarui bobotnya setiap batch (8000 gambar pelatihan dibagi ukuran batch 32).
  - `epochs=50`: Model dilatih selama 50 epoch.
  - `validation_data=test_set`: Dataset validasi.
  - `validation_steps=2000/32`: 2000 gambar validasi dibagi ukuran batch 32.


# Output
Setelah model dilatih, ia akan memprediksi klasifikasi biner (misalnya, "positif" atau "negatif") untuk gambar input berdasarkan dataset pelatihan dan validasi.
