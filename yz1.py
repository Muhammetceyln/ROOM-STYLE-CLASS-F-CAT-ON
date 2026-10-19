import torch

# CUDA yoksa işlemciyi kullan
device = torch.device("cpu") 
print(f"Eğitim şu cihazda yapılacak: {device}")

# 1. Model: Tam Bağlantılı Sinir Ağı (FCNN)
import tensorflow as tf
model_1 = tf.keras.Sequential([
    # Ön İşleme: Piksel değerlerini 0-255 arasından 0-1 arasına çekiyoruz. Bilgisayar değerleri anlamlandırıyor bu sayede
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # 2 Boyutlu resmi tek boyutlu düz bir listeye çeviriyoruz
    tf.keras.layers.Flatten(),
    
    # 5 Katmanlı yapı:
    tf.keras.layers.Dense(512, activation='relu'),  # 1. Katman
    tf.keras.layers.Dense(256, activation='relu'),  # 2. Katman
    tf.keras.layers.Dense(128, activation='relu'),  # 3. Katman
    tf.keras.layers.Dense(64, activation='relu'),   # 4. Katman
    tf.keras.layers.Dense(4, activation='softmax')  # 5. Çıkış Katmanı (4 farklı oda tarzımız olduğu için 4 nöron)
])

# Modeli öğrenmeye hazır hale getiriyoruz
model_1.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Modelin mimarisini özet olarak görelim
model_1.summary()

# Eğitimi başlatalım (10 Epoch - Tur -)
print("\n--- 1. Model Eğitimi Başlıyor ---\n")
history_1 = model_1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 2. Model: Temel Evrişimsel Sinir Ağı (CNN)
import tensorflow as tf

model_2 = tf.keras.Sequential([
    # Uyarıyı gidermek için Input katmanı ekliyoruz
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1./255),
    
    # 1. Evrişim (Conv) Katmanı
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 2. Evrişim Katmanı
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 3. Evrişim Katmanı
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 4. Evrişim Katmanı
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 5. Evrişim Katmanı
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Öğrenilen özellikleri sınıflandırmak için düzleştirip çıkışa veriyoruz
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax') # 4 Sınıf
])

model_2.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model_2.summary()

print("\n--- 2. Model Eğitimi Başlıyor ---\n")
history_2 = model_2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 3. Model: Düzenleme (L2 Regularization) Eklenmiş CNN
import tensorflow as tf
from tensorflow.keras import regularizers

model_3 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    
    # İŞTE BURASI FARKLI: L2 Düzenlemesi (Regularization) ekledik
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    
    tf.keras.layers.Dense(4, activation='softmax')
])

model_3.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\n--- 3. Model Eğitimi Başlıyor (L2 Regularization) ---\n")
history_3 = model_3.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
# 4. Model: Dropout Eklenmiş CNN
import tensorflow as tf

model_4 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    
    # İŞTE BURASI FARKLI: Karar vermeden önce nöronların %50'sini rastgele kapatıyoruz
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(4, activation='softmax')
])

model_4.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\n--- 4. Model Eğitimi Başlıyor (Dropout) ---\n")
history_4 = model_4.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
# 5. Model: Batch Normalization Eklenmiş CNN
import tensorflow as tf

model_5 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(), # İŞTE BURASI YENİ
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(), # Dense katmanına da ekleyebiliriz
    
    tf.keras.layers.Dense(4, activation='softmax')
])

model_5.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\n--- 5. Model Eğitimi Başlıyor (Batch Normalization) ---\n")
history_5 = model_5.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
# 6. Model: Veri Artırma (Data Augmentation) Eklenmiş CNN
import tensorflow as tf

# Veri Artırma Katmanlarını hazırlayalım
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"), # Rastgele yatay çevir (Ayna görüntüsü)
  tf.keras.layers.RandomRotation(0.1),      # Rastgele %10 sağa/sola yatır
  tf.keras.layers.RandomZoom(0.1),          # Rastgele %10 yakınlaştır veya uzaklaştır
])

model_6 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    
    # 1. Aşama: Veriyi Artır ve Normalize et
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    
    # Klasik 5 Katmanlı CNN (Saf haliyle, hocanın istediği gibi)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model_6.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\n--- 6. Model Eğitimi Başlıyor (Data Augmentation) ---\n")
history_6 = model_6.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

