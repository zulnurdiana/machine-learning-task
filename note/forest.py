from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Employee.csv')
print(df.head())
print(df.shape)

X = df
y = df['LeaveOrNot']

# Preprocess
X = pd.get_dummies(X)

# PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
print(pca.explained_variance_ratio_)

# Split data
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=30)

# Gunakan Random Forest Classifier
model = RandomForestClassifier()

# Latih model
model.fit(X_train_pca, y_train)

# Evaluasi model
print(model.score(X_test_pca, y_test))
