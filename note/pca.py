import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Memuat hanya 10 data pertama dari dataset
df = pd.read_csv('Employee.csv')

print(df.head(10))

# Memilih fitur yang akan digunakan untuk PCA
features = ['Gender', 'PaymentTier', 'Age',
            'ExperienceInCurrentDomain', "LeaveOrNot"]

X = df[features]

# Mengubah variabel kategorikal menjadi variabel dummy
X = pd.get_dummies(X)

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah komponen utama yang diinginkan untuk PCA
n_components = 2
pca = PCA(n_components)

# Melakukan PCA pada data yang sudah dinormalisasi
X_pca = pca.fit_transform(X_scaled)

print(X_pca)

# Visualisasi hasil PCA
plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Gender'].map(
#     {'Male': 'red', 'Female': 'blue'}))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['LeaveOrNot'], cmap='viridis')
# plt.colorbar(label='LeaveOrNot')
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Analysis')
plt.show()

