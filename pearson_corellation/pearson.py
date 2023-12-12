import pandas as pd


# Membaca dataset
df = pd.read_csv('Employee.csv')
col1 = "Age"
col2 = "JoiningYear"

print(df.head(10))

# Menghitung korelasi Pearson antara Kolom1 dan Kolom2
correlation = df[col1].corr(df[col2])

lower_bound = -1
upper_bound = 1


print("Nilai korelasi antara kolom", col1, "dan", col2, "adalah", correlation)

#  Pengecekan nilai hasil korelasi
if correlation == 0.0:
    print("Tidak ada keterkaitan antar 2 variabel")
elif lower_bound < correlation < upper_bound:
    print("Tidak ada keterkaitan yang signifikan antara kolom ini")
else:
    print("Terdapat keterkaitan yang signifikan antara kolom ini")
