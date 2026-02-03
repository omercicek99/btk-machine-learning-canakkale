import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

dir_path=os.path.dirname(os.path.realpath('__file__'))
data=pd.read_csv(dir_path+"/ObesityDataSet_raw_and_data_sinthetic.csv")
print(data.head(5))


# Veri kümesindeki öznitelik isimlerini Türkçe'ye çeviriyoruz
data = data.rename(columns={
    'Gender': 'Cinsiyet',
    'Age': 'Yas',
    'Height': 'Boy',
    'Weight': 'Kilo',
    'family_history_with_overweight': 'obezite_aile_gecmisi',
    'FAVC': 'yuksek_kalorili_gida',
    'FCVC': 'sebze_tuketimi',
    'NCP': 'ana_ogun_sayisi',
    'CAEC': 'ara_ogun',
    'SMOKE': 'sigara',
    'CH2O': 'su_tuketimi',
    'SCC': 'kalori_takibi',
    'FAF': 'fiziksel_aktivite',
    'TUE': 'teknoloji_kullanimi',
    'CALC': 'alkol',
    'MTRANS': 'ulasim_turu'
})

# Özniteliklerimizi sınıflandırma algoritmaları için sayısal hale getiriyoruz
labelEncoder = LabelEncoder()

data['Cinsiyet'] = labelEncoder.fit_transform(data['Cinsiyet'])
data['obezite_aile_gecmisi'] = labelEncoder.fit_transform(data['obezite_aile_gecmisi'])
data['yuksek_kalorili_gida'] = labelEncoder.fit_transform(data['yuksek_kalorili_gida'])
data['ara_ogun'] = labelEncoder.fit_transform(data['ara_ogun'])
data['sigara'] = labelEncoder.fit_transform(data['sigara'])
data['kalori_takibi'] = labelEncoder.fit_transform(data['kalori_takibi'])
data['alkol'] = labelEncoder.fit_transform(data['alkol'])
data['ulasim_turu'] = labelEncoder.fit_transform(data['ulasim_turu'])

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Veri kümesinde bulunan bağımlı(sınıf) değişkeni ve test kümesi boyutunu %30 olacak şekilde belirliyoruz
y = data.NObeyesdad.values
x = data.drop('NObeyesdad', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Sınıflandırma algoritmasının kurulumunu yapıyoruz. k=5 ve uzaklık ölçüsü öklid seçilmiştir.
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

# Veri kümemizde bulunan bağımlı(sınıf) değişkeni ve test kümesi boyutunu %30 olacak şekilde belirliyoruz.
y = data.NObeyesdad.values
x = data.drop('NObeyesdad', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Sınıflandırma algoritmasının kurulumunu yapıyoruz. k=5 ve uzaklık ölçüsü öklid seçilmiştir.
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))

print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(
          knn.n_neighbors,
          knn.score(x_test, y_test),
          precision_score(y_test, prediction, average='weighted'),
          recall_score(y_test, prediction, average='weighted'),
          f1_score(y_test, prediction, average='weighted')
      ))

knnscore = {}

for i in range(3, 100, 2):
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    knn.fit(x_train, y_train)
    knnscore[i] = knn.score(x_test, y_test)

knnscore

best_k = max(knnscore, key=knnscore.get)
print("En iyi k:", best_k, "Skor:", knnscore[best_k])


plt.plot(knnscore.keys(),knnscore.values())
plt.xlabel("k değeri")
plt.ylabel("doğruluk")
plt.show()

prediction=knn.predict([[0,40,1.74,79,1,0,3.0,1.0,2,1,2.0,1,0.0,1,3,3]])
print(prediction)
