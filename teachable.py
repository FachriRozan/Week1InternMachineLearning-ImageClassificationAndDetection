import tensorflow as tf
import numpy as np
import cv2
import os

# Fungsi untuk memuat model .pb
def load_pb_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

# Fungsi untuk memuat dan mempersiapkan gambar
def load_and_preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Fungsi untuk melakukan inferensi menggunakan model
def test_image(model, image):
    # Konversi gambar ke tensor float32
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    # Mengambil fungsi inferensi default dari model
    infer = model.signatures['serving_default']
    # Melakukan inferensi
    output = infer(input_tensor)
    return output

# Fungsi untuk menginterpretasi hasil inferensi
def interpret_output(output, class_names):
    # Menyesuaikan nama output layer
    output_tensor = list(output.values())[0]  # Ambil tensor output pertama
    predictions = output_tensor.numpy()
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    predicted_confidence = predictions[0][predicted_class_index] * 100
    return predicted_class_name, predicted_confidence

# Path ke folder model dan gambar uji
model_path = 'tes/model.savedmodel'
image_path = 'kucing.jpeg'
image_size = (224, 224)  # Ukuran input model (sesuaikan dengan model Anda)
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike']  # Daftar nama kelas

# Memuat model
model = load_pb_model(model_path)

# Memuat dan mempersiapkan gambar uji
image = load_and_preprocess_image(image_path, image_size)

# Melakukan inferensi
output = test_image(model, image)

# Menginterpretasi hasil inferensi
predicted_class_name, predicted_confidence = interpret_output(output, class_names)

# Menampilkan hasil
print(f"Predicted class: {predicted_class_name} with confidence: {predicted_confidence:.2f}%")
