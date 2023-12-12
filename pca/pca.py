import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Memuat hanya 10 data pertama dari dataset
df = pd.read_csv('Employee.csv')
print(df.head())
print(df.shape)
X = df

# Variabel yang dijadikan target untuk PCA
y = df['LeaveOrNot']

# Memberikan variabel kategorikal menjadi variabel dummy
X = pd.get_dummies(X)

# Menggunakan 95 % variansi data asli
pca = PCA(0.95)

# Melakukan PCA untuk mereduksi dimensi
X_pca = pca.fit_transform(X)

print(X_pca.shape)

print(pca.explained_variance_ratio_)

# Memecah data menjadi training dan testing
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=30)

# Membuat model
model = RandomForestClassifier()

# Melatih model
model.fit(X_train_pca, y_train)

# Melihat akurasi
print(model.score(X_test_pca, y_test))
