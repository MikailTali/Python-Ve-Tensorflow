import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf
# Verimizi getirdik
dataFrame = pd.read_excel('merc.xlsx')
print(dataFrame.head())

# Veriyi Anlamak , elimizde ne var görmek gerekir.
print(dataFrame.describe())
print(dataFrame.isnull().sum())

# Grafiksel Analiz
sbn.histplot(dataFrame["price"])
plt.show()

# Sayısını göster
sbn.countplot(dataFrame["year"])
plt.show()

# Korelasyon
print(dataFrame.corr()["price"].sort_values())
# Seaborn ile Dağılım Grafiği Oluşturma
sbn.scatterplot(x="mileage", y="price" , data =dataFrame)
plt.show()
# Verileri Sıralama ve Görüntüleme
print(dataFrame.sort_values("price" , ascending = False).head(20)) 
print(dataFrame.sort_values("price" , ascending = True).head(20)) 
"""# Veri Temizliği"""
# Belirli Bir Aralıktaki Verileri Seçme
dataFrame1=dataFrame.sort_values('price' , ascending= False).iloc[131:]
# İstatistiksel Özeti Yazdırma
print(dataFrame1.describe())

# Veri Dağılım Grafiği Oluşturma
plt.figure(figsize=(7,5))
sbn.displot(dataFrame1["price"])
plt.show()

# Yıla Göre Ortalama Fiyatı Hesaplama
dataFrame.groupby("year").mean()["price"]
dataFrame1.groupby("year").mean()["price"]

# Belirli Bir Yıl Dışında Verileri Filtreleme
dataFrame = dataFrame[dataFrame.year != 1970]

# Yıl Bazında Gruplama
dataFrame.groupby("year")
print(dataFrame.head())

# transmission Sütununun Kaldırılması
dataFrame = dataFrame.drop("transmission" , axis=1)
print(dataFrame)

# Hedef Değişkenin ve Özelliklerin Ayrılması
y = dataFrame['price'].values
x = dataFrame.drop('price' , axis = 1).values

# Verilerin Eğitim ve Test Setlerine Bölünmesi
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=10)
print(len(x_train))
print(len(y_train))

# Verilerin Ölçeklenmesi
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# TensorFlow ve Keras ile Model Oluşturma
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # Dense katmanlarını içe aktarma
import tensorflow as tf  # TensorFlow ana kütüphanesini içe aktarma

# Eğitim verisinin şeklini yazdırma
print(x_train.shape)

# Model oluşturma , Katman Eklemeleri , Sequential sınıfı kullanılarak bir model oluşturma
model = Sequential()
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))

# Modelin Derlenmesi
model.compile(optimizer='adam', loss='mse')

# Modelin Eğitilmesi
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300)

# Kayıp Verisinin Görselleştirilmesi
kayipVerisi = pd.DataFrame(model.history.history)
kayipVerisi.plot()
plt.show()

# Tahmin ve Hata Hesaplama
from sklearn.metrics import mean_squared_error , mean_absolute_error
tahminDizisi = model.predict(x_test)
mean_absolute_error(y_test , tahminDizisi)

# Gerçek ve Tahmin Değerlerinin Karşılaştırılması
plt.scatter(y_test , tahminDizisi)
plt.plot(y_test,y_test,'g-*')

# Yeni Veri ile Tahmin Yapma
dataFrame.iloc[2]
yeniArabaSeries = dataFrame.drop('price' , axis=1).iloc[2]
type(yeniArabaSeries)
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshae(-1,5))
model.predict(yeniArabaSeries)