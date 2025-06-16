# %% [markdown]
# # **1. Perkenalan Dataset**

# %% [markdown]
# Dataset yang digunakan dalam penelitian ini diambil dari portal resmi United Nations Sustainable Development Goals (UN SDGs) yang tersedia di https://unstats.un.org/sdgs/dataportal. Dataset ini berfokus pada Goal 4: Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all, yang bertujuan untuk memastikan pendidikan berkualitas dan inklusif serta mendorong kesempatan belajar sepanjang hayat bagi semua orang.          
# Data yang digunakan mencakup berbagai indikator pendidikan yang relevan dengan target-target dari Goal 4, seperti proporsi anak-anak dan pemuda yang mencapai tingkat kemahiran minimum dalam membaca dan matematika. Dataset ini memuat informasi dalam bentuk waktu, wilayah geografis, indikator spesifik, serta berbagai atribut terkait seperti jenis kelamin, tingkat pendidikan, dan jenis keterampilan (misalnya membaca dan matematika).                 
# Beberapa kolom penting dalam dataset ini antara lain:             
# - **Goal, Target, Indicator:** Menjelaskan tujuan dan target pendidikan sesuai SDGs.
# - **GeoAreaName:** Nama negara atau wilayah, yang dalam studi ini difokuskan pada negara-negara dengan populasi mayoritas Muslim.
# - **TimePeriod:** Tahun pengumpulan data.
# - **Value:** Nilai indikator pendidikan yang diukur, misalnya persentase pencapaian kemahiran membaca atau matematika.
# - **Sex, Education level, Type of skill:** Keterangan demografis dan jenis keterampilan yang diuji.
# - **Source:** Sumber data, misalnya dari Programme for International Student Assessment (PISA).

# %% [markdown]
# # **2. Import Library**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib

# %% [markdown]
# # **3. Memuat Dataset**

# %%
df = pd.read_csv('Goal4.csv')
df.head()

# %% [markdown]
# # **4. Exploratory Data Analysis (EDA)**

# %% [markdown]
# ## Struktur Data

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ## Missing Value

# %%
print(df.isnull().sum())

# %% [markdown]
# ## Duplikat Value

# %%
print("Jumlah data duplikat:", df.duplicated().sum())


# %% [markdown]
# ## Analisis Distribusi dan Korelasi

# %%
# Pilih hanya kolom numerik
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(16, 10))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns)//4 + 1, 4, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribusi {col}')

plt.tight_layout()
plt.show()

# %%
# Ambil semua kolom kategorikal (tipe objek)
categorical_columns = df.select_dtypes(include='object').columns

# Ukuran grid
n_cols = 3
n_rows = -(-len(categorical_columns) // n_cols)  # Pembulatan ke atas

# Buat figure besar
plt.figure(figsize=(n_cols * 6, n_rows * 4))

# Plot tiap kolom
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    
    # Batasi hanya 20 kategori teratas
    top_categories = df[col].value_counts().nlargest(20).index
    filtered_df = df[df[col].isin(top_categories)]
    
    sns.countplot(data=filtered_df, x=col, order=top_categories, palette='viridis')
    plt.title(col, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.xlabel('')
    plt.ylabel('Count')
    
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.suptitle("Distribusi Kolom Kategorikal (Top 20)", fontsize=18)
plt.show()

# %% [markdown]
# # **5. Data Preprocessing**

# %% [markdown]
# ## Menghapus Kolom Yang Tidak Penting

# %%
columns_to_drop = [
    'TimeCoverage',
    'UpperBound',
    'LowerBound',
    'GeoInfoUrl',
    'FootNote',
    'BasePeriod',
    'SeriesCode',
    'GeoAreaCode',
    'Time_Detail',
    'Goal',
    'Target',
    'Indicator',
    'Source',
    'Reporting Type'
]
df = df.drop(columns=columns_to_drop)


# %%
df = df.dropna(subset=['Education level'])  # Jika ingin langsung mengubah df utama


# %%
df.info()

# %% [markdown]
# ## Missing Value

# %%
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing,
    'Percentage (%)': missing_percent
})

print(missing_df)


# %%
df.drop(['Quantile', 'Age', 'Type of skill', 'Location'], axis=1, inplace=True)


# %%
df['Sex'].fillna(df['Sex'].mode()[0], inplace=True)


# %%
df.info()

# %%
df.head()

# %% [markdown]
# ## Preprocessing Outlier

# %%
# Pilih hanya kolom numerik
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# %%
numerical_columns = ['TimePeriod', 'Value']  # hanya kolom numerik

for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"{col}: {len(outliers)} outliers ({round(100*len(outliers)/len(df), 2)}%)")



# %% [markdown]
# ## Preprocessing Encoding

# %%
# Pilih hanya kolom bertipe object (kategorikal)
categorical_columns = df.select_dtypes(include=['object'])

# Hitung jumlah kategori unik di setiap kolom kategorikal
category_counts = categorical_columns.nunique()

# Tampilkan hasil
print("Jumlah Kategori di Kolom Kategorikal:\n", category_counts)

# %%
edu_counts = df['Education level'].value_counts().reset_index()
edu_counts.columns = ['Education level', 'Count']
print(edu_counts)


# %% [markdown]
# # **7. Pembangunan Model CLuster**

# %% [markdown]
# ## **Kmeans**

# %%
df_kmeans = df.copy()

# %% [markdown]
# ### Encoding

# %%
# Kolom-kolom kategorikal yang akan diencode
columns_to_encode = ['SeriesDescription', 'GeoAreaName', 'Education level', 'Nature', 'Sex', 'Units']

# Simpan encoder-nya supaya bisa decode nanti
label_encoders_kmeans = {}

for col in columns_to_encode:
    le = LabelEncoder()
    df_kmeans[col] = le.fit_transform(df_kmeans[col])
    label_encoders_kmeans[col] = le  # simpan encoder untuk decode nanti


# %%
df_kmeans.head()

# %% [markdown]
# ### Standarisasi

# %%
scaler_kmeans = StandardScaler()
X_scaled_kmeans = scaler_kmeans.fit_transform(df_kmeans)

# %%
scores_kmeans = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled_kmeans)
    score_kmeans = silhouette_score(X_scaled_kmeans, labels_kmeans)
    scores_kmeans.append(score_kmeans)  # Append skor ke list

plt.plot(range(2, 11), scores_kmeans, marker='o')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score untuk KMeans")
plt.grid(True)
plt.show()


# %%
pca_kmeans = PCA(n_components=2)
X_pca_kmeans = pca_kmeans.fit_transform(X_scaled_kmeans)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_kmeans[:, 0], X_pca_kmeans[:, 1], alpha=0.5)
plt.title("PCA - Sebelum Clustering")   
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# %%
# Asumsikan kamu sudah punya hasil PCA (X_pca) dan labels dari KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled_kmeans)

# %%
# Visualisasi hasil cluster
plt.figure(figsize=(7, 5))
scatter_kmeans = plt.scatter(X_pca_kmeans[:, 0], X_pca_kmeans[:, 1], c=labels_kmeans, cmap='Set1', s=10)
plt.title('Hasil Clustering dengan KMeans (PCA Projection)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar(scatter_kmeans, label='Cluster')
plt.grid(True)
plt.show()


# %%
# Hitung silhouette score
scores_kmeans = silhouette_score(X_scaled_kmeans, labels_kmeans)

print(f"Silhouette Score untuk KMeans (k=3): {scores_kmeans:.4f}")

# %% [markdown]
# ## **Gaussian Mixture Model (GMM)**

# %%
categorical_cols = ['SeriesDescription', 'GeoAreaName', 'Education level', 'Nature', 'Sex', 'Units']
numerical_cols = ['TimePeriod', 'Value']

# %%
df_GMM = df.copy()

# %% [markdown]
# ### Encoding

# %%
ohe_GMM = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_GMM = ohe_GMM.fit_transform(df_GMM[categorical_cols])
joblib.dump(ohe_GMM, 'encoder.joblib')  # Simpan encoder untuk nanti decode

# %%
# Simpan nama-nama fitur hasil one-hot
ohe_feature_names_GMM = ohe_GMM.get_feature_names_out(categorical_cols)

# %%
X_num_GMM = df_GMM[numerical_cols].values
X_combined_GMM = np.hstack((X_cat_GMM, X_num_GMM))

# %%
scaler_GMM = StandardScaler()
X_scaled_GMM = scaler_GMM.fit_transform(X_combined_GMM)

# %%
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels_GMM = gmm.fit_predict(X_scaled_GMM)
    score_GMM = silhouette_score(X_scaled_GMM, labels_GMM)
    print(f"GMM k={k} → Silhouette Score: {score_GMM:.4f}")

# %%
gmm = GaussianMixture(n_components=2, random_state=42)
labels_GMM = gmm.fit_predict(X_scaled_GMM)
score_GMM = silhouette_score(X_scaled_GMM, labels_GMM)
print(f"GMM k={2} → Silhouette Score: {score_GMM:.4f}")

# %%
# === 1. Hitung Silhouette Score ===
score_GMM = silhouette_score(X_scaled_GMM, labels_GMM)
print(f"Silhouette Score untuk GMM: {score_GMM:.4f}")

# === 2. Reduksi dimensi ke 2D (agar bisa divisualisasikan) ===
pca_GMM = PCA(n_components=2, random_state=42)
X_pca_GMM = pca_GMM.fit_transform(X_scaled_GMM)

# === 3. Visualisasi hasil cluster ===
plt.figure(figsize=(8,6))
scatter_GMM = plt.scatter(X_pca_GMM[:, 0], X_pca_GMM[:, 1], c=labels_GMM, cmap='Set2', s=40, alpha=0.7)
plt.title(f'Clustering GMM (PCA 2D)\nSilhouette Score = {score_GMM:.4f}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter_GMM, label='Cluster Label')
plt.grid(True)
plt.show()

# %% [markdown]
# ## **Agglomerative Hierarchy**

# %% [markdown]
# ### One Hot Encoding

# %%
# Salin data
df_copy = df.copy()

# Inisialisasi encoder dan fit-transform
ohe = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
X_cat = ohe.fit_transform(df_copy[categorical_cols])

# Ambil nama-nama kolom hasil encoding
encoded_cat_cols = ohe.get_feature_names_out(categorical_cols)

# Ambil kolom numerik
X_num = df_copy[numerical_cols].values

# Gabungkan semua data
X_all = np.hstack([X_cat, X_num])

# Gabungkan ke dalam DataFrame
df_Hierarchy = pd.DataFrame(X_all, columns=list(encoded_cat_cols) + numerical_cols)

# %% [markdown]
# ### Standarisasi

# %%
scaler_Hierarchy = StandardScaler()
X_scaled_Hierarchy = scaler_Hierarchy.fit_transform(df_Hierarchy)

# %%
# PCA ke 2 komponen
pca_Hierarchy = PCA(n_components=2)
X_pca_Hierarchy = pca_Hierarchy.fit_transform(X_scaled_Hierarchy)

# Agglomerative Clustering dengan 3 cluster (bisa diganti)
agglo_Hierarchy = AgglomerativeClustering(n_clusters=3)
labels_agglo_Hierarchy = agglo_Hierarchy.fit_predict(X_pca_Hierarchy)

# Simpan hasil cluster ke df_hirarki
df_Hierarchy['Cluster'] = labels_agglo_Hierarchy + 1

# Hitung Silhouette Score
score_Hierarchy = silhouette_score(X_pca_Hierarchy, labels_agglo_Hierarchy)
print(f"Silhouette Score untuk Agglomerative Clustering: {score_Hierarchy:.4f}")

# Visualisasi hasil cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_Hierarchy[:, 0], y=X_pca_Hierarchy[:, 1], hue=labels_agglo_Hierarchy, palette='Set2', s=40)
plt.title(f'Agglomerative Clustering (PCA 2D)\nSilhouette Score = {score_Hierarchy:.4f}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()


# %% [markdown]
# ## **8. Analisis dan Interpretasi Hasil Cluster**

# %% [markdown]
# ## Interpretasi Target

# %%
df_Hierarchy.info()

# %%
df_Hierarchy.head()

# %% [markdown]
# ## Inversi Data

# %%
# Inversi OneHotEncoder
decoded_cats = ohe.inverse_transform(X_cat)
df_decoded_cats = pd.DataFrame(decoded_cats, columns=categorical_cols, index=df_Hierarchy.index)

# Reset index agar benar-benar clean
df.reset_index(drop=True, inplace=True)

# Gabung hasil inverse + numerik
df_original = pd.concat([df_decoded_cats, df[numerical_cols]], axis=1)

# Reset index untuk jaga-jaga
df_original.reset_index(drop=True, inplace=True)
df_Hierarchy.reset_index(drop=True, inplace=True)


print(len(df_original))         # Harus 13848
print(len(df_Hierarchy))        # Harus 13848
print(len(df_Hierarchy['Cluster']))  # Harus 13848

# %%
print(df_original.info())

# %%
# Gabungkan Cluster
df_original['Cluster'] = df_Hierarchy['Cluster'].values

# %%
print(df_original.head())

# %% [markdown]
# ## **Insight Data**

# %% [markdown]
# ### **1. Rata-rata dan sebaran nilai untuk tiap cluster**

# %%
print(df_original.groupby("Cluster")["Value"].describe())

# %% [markdown]
# ### **2. Komposisi kategori per cluster**

# %%
print(pd.crosstab(df_original["Cluster"], df_original["Education level"], normalize='index') * 100)


# %%
print(pd.crosstab(df_original["Cluster"], df_original["Sex"], normalize='index') * 100)

# %%
print(pd.crosstab(df_original["Cluster"], df_original["Value"], normalize='index') * 100)

# %%
print(pd.crosstab(df_original["Cluster"], df_original["Nature"], normalize='index') * 100)

# %%
print(pd.crosstab(df_original["Cluster"], df_original["TimePeriod"], normalize='index') * 100)


# %% [markdown]
# ### **3. SeriesDescription per Cluster**

# %%
print(df_original.groupby("Cluster")["SeriesDescription"].agg(lambda x: x.value_counts().head(3)))

# %% [markdown]
# ### **4. Visualisasi per cluster**

# %%
sns.histplot(data=df_original, x="Value", hue="Cluster", element="step", stat="density", common_norm=False)
plt.title("Distribusi Nilai per Cluster")
plt.show()

# %%
sns.boxplot(data=df_original, x="Cluster", y="Value")
plt.title("Boxplot Nilai Tiap Cluster")
plt.show()


# %% [markdown]
# ### **5. Ringkasan Karakteristik**

# %%
summary = df_original.groupby("Cluster").agg({
    "Value": "mean",
    "Education level": lambda x: x.value_counts().idxmax(),
    "Sex": lambda x: x.value_counts().idxmax(),
    "TimePeriod": lambda x: x.value_counts().idxmax()
})
summary.rename(columns={"Value": "Rata-rata Value"}, inplace=True)
print(summary)


# %% [markdown]
# # **Analisis Karakteristik Cluster dari Model Agglomerative Hierarchical Clustering**
# 
# Berikut adalah analisis karakteristik untuk masing-masing cluster hasil dari model Agglomerative Hierarchical Clustering berdasarkan atribut pendidikan, jenis kelamin, dan waktu.
# 
# ---
# ## **Cluster 1**:
# - **Rata-rata nilai indikator:** 77.67
# - **Tingkat pendidikan dominan:** PRIMAR (pendidikan dasar)
# - **Jenis kelamin dominan:** BOTHSEX
# - **Tahun dominan:** 2022
# - **Distribusi nilai:** Tinggi di kisaran 90–100%, penyebaran sedang hingga luas
# - **Analisis:**
# Cluster ini mencerminkan negara atau wilayah dengan capaian pendidikan dasar yang sangat baik. Nilai-nilai indikator pendidikan tinggi dan stabil menunjukkan sistem pendidikan yang sudah matang. Bisa diasosiasikan dengan negara-negara dengan kebijakan pendidikan dasar universal dan efektif.
# 
# ## **Cluster 2**:
# - **Rata-rata nilai indikator:** 0.80 (sangat rendah)
# - **Tingkat pendidikan dominan:** LOWSEC (pendidikan menengah bawah)
# - **Jenis kelamin dominan:** BOTHSEX
# - **Tahun dominan:** 2019 
# - **Distribusi nilai:** Sangat sempit di nilai mendekati nol
# - **Analisis:**
# Cluster ini merepresentasikan kelompok dengan capaian pendidikan yang sangat rendah, khususnya di jenjang menengah bawah. Rendahnya nilai dan sebaran yang sempit mengindikasikan konsistensi kondisi buruk atau data dari area yang tertinggal. Bisa juga mencerminkan kekurangan dalam pelaporan data pendidikan.
# 
# ## **Cluster 3**:
# - **Rata-rata nilai indikator:** 59.57
# - **Tingkat pendidikan dominan:** LOWSEC
# - **Jenis kelamin dominan:** BOTHSEX
# - **Tahun dominan: 2017**
# - **Distribusi nilai:** Luas, dari mendekati nol hingga hampir 100%
# - **Analisis:**
# Cluster ini menggambarkan wilayah yang berada dalam proses peningkatan capaian pendidikan, khususnya pada tingkat menengah bawah. Sebaran nilai yang luas menandakan heterogenitas: baik negara yang sedang berkembang maupun beberapa yang telah mencapai kemajuan sedang–tinggi dalam pendidikan.

# %% [markdown]
# # **9. Mengeksport Data**
# 

# %%
df_original.to_csv("Data_Clustering.csv", index=False)


