import pandas as pd
from scipy.stats import chi2_contingency

# Membaca dataset
df = pd.read_csv('Employee.csv')

# Membuat tabel kontingensi untuk menghitung frekuensi kemunculan setiap kombinasi nilai antara dua variabel.
contingency_table = pd.crosstab(
    df['Education'], df['Gender'])


# Menampilkan tabel kontingensi
print("Tabel Kontingensi:")
print(contingency_table)


# Melakukan uji Chi-Square
chi2, _, _, expected = chi2_contingency(contingency_table)

# Menampilkan hasil uji Chi-Square
print("\nHasil Uji Chi-Square:")

# Nilai X2 nya
print(f"Chi-Square Statistic: {chi2}")

# matriks yang berisi frekuensi yang diharapkan untuk setiap sel dalam tabel kontingensi
print(f"Expected Frequencies: {expected}")
