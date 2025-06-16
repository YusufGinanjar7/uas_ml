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
import streamlit as st
import plotly.graph_objects as go

# %% [markdown]
# # **3. Streamlit App Configuration**

st.set_page_config(page_title="SDGs Goal 4 Clustering", layout="wide")
st.title("🎓 Clustering Goal 4 (Pendidikan) - SDGs")
st.write("Hasil clustering dan visualisasi analisis SDGs menggunakan data Goal 4.")

# Sidebar untuk navigasi
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.selectbox("Pilih Bagian:", 
    ["Dataset Overview", "Data Preprocessing", "Clustering Analysis", "Results & Insights"])

# %% [markdown]
# # **4. Memuat Dataset**

@st.cache_data
def load_data():
    df = pd.read_csv('Goal4.csv')
    return df

if st.sidebar.button("🔄 Load Data"):
    df = load_data()
    st.session_state.df = df

if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# %% [markdown]
# # **5. Dataset Overview Section**

if menu == "Dataset Overview":
    st.header("📋 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Informasi Dataset")
        st.write(f"**Jumlah Baris:** {df.shape[0]:,}")
        st.write(f"**Jumlah Kolom:** {df.shape[1]}")
        st.write(f"**Ukuran Dataset:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.subheader("🔍 Sample Data")
        st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("📈 Struktur Data")
    buffer = []
    for col in df.columns:
        buffer.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null %': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%"
        })
    
    info_df = pd.DataFrame(buffer)
    st.dataframe(info_df, use_container_width=True)
    
    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing Values Visualization
    st.subheader("❌ Visualisasi Missing Values")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        fig_missing = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title="Missing Values per Column",
            labels={'x': 'Count', 'y': 'Columns'}
        )
        fig_missing.update_layout(height=500)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("✅ Tidak ada missing values!")
    
    # Duplicate Values
    st.subheader("🔄 Duplicate Values")
    duplicates = df.duplicated().sum()
    st.metric("Jumlah Data Duplikat", duplicates)
    
    # Distribution Analysis
    st.subheader("📊 Analisis Distribusi")
    
    # Numerical columns distribution
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_columns) > 0:
        st.write("**Distribusi Kolom Numerik:**")
        
        fig_num = plt.figure(figsize=(16, 10))
        for i, col in enumerate(numerical_columns, 1):
            plt.subplot(len(numerical_columns)//2 + 1, 2, i)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribusi {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig_num)
    
    # Categorical columns distribution
    categorical_columns = df.select_dtypes(include='object').columns
    if len(categorical_columns) > 0:
        st.write("**Distribusi Kolom Kategorikal:**")
        
        selected_cat = st.selectbox("Pilih kolom kategorikal:", categorical_columns)
        
        top_categories = df[selected_cat].value_counts().head(20)
        
        fig_cat = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            title=f'Top 20 Kategori - {selected_cat}',
            labels={'x': 'Count', 'y': selected_cat}
        )
        fig_cat.update_layout(height=600)
        st.plotly_chart(fig_cat, use_container_width=True)

# %% [markdown]
# # **6. Data Preprocessing Section**

elif menu == "Data Preprocessing":
    st.header("🔧 Data Preprocessing")
    
    # Create preprocessing steps
    st.subheader("1️⃣ Menghapus Kolom Yang Tidak Penting")
    
    columns_to_drop = [
        'TimeCoverage', 'UpperBound', 'LowerBound', 'GeoInfoUrl', 'FootNote',
        'BasePeriod', 'SeriesCode', 'GeoAreaCode', 'Time_Detail', 'Goal',
        'Target', 'Indicator', 'Source', 'Reporting Type'
    ]
    
    # Show columns that will be dropped
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_cols_to_drop:
        st.write("**Kolom yang akan dihapus:**")
        for col in existing_cols_to_drop:
            st.write(f"- {col}")
        
        # Apply preprocessing
        df_processed = df.drop(columns=existing_cols_to_drop)
        df_processed = df_processed.dropna(subset=['Education level'])
        
        st.success(f"✅ Berhasil menghapus {len(existing_cols_to_drop)} kolom")
        st.write(f"**Shape setelah preprocessing:** {df_processed.shape}")
    else:
        df_processed = df.copy()
        st.info("ℹ️ Tidak ada kolom yang perlu dihapus")
    
    st.subheader("2️⃣ Handling Missing Values")
    
    # Remove additional columns
    additional_drops = ['Quantile', 'Age', 'Type of skill', 'Location']
    existing_additional = [col for col in additional_drops if col in df_processed.columns]
    
    if existing_additional:
        df_processed = df_processed.drop(columns=existing_additional)
        st.write(f"**Menghapus kolom tambahan:** {existing_additional}")
    
    # Fill missing values
    if 'Sex' in df_processed.columns and df_processed['Sex'].isnull().sum() > 0:
        df_processed['Sex'].fillna(df_processed['Sex'].mode()[0], inplace=True)
        st.write("✅ Missing values pada kolom 'Sex' telah diisi dengan modus")
    
    # Show final structure
    st.subheader("3️⃣ Hasil Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Struktur Data Akhir:**")
        st.write(f"- Jumlah Baris: {df_processed.shape[0]:,}")
        st.write(f"- Jumlah Kolom: {df_processed.shape[1]}")
        st.write(f"- Total Missing Values: {df_processed.isnull().sum().sum()}")
    
    with col2:
        st.write("**Sample Data Setelah Preprocessing:**")
        st.dataframe(df_processed.head(3), use_container_width=True)
    
    # Outlier Analysis
    st.subheader("4️⃣ Analisis Outlier")
    
    numerical_columns = ['TimePeriod', 'Value']
    outlier_info = []
    
    for col in numerical_columns:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
            
            outlier_info.append({
                'Column': col,
                'Outliers Count': len(outliers),
                'Percentage': f"{round(100*len(outliers)/len(df_processed), 2)}%"
            })
    
    outlier_df = pd.DataFrame(outlier_info)
    st.dataframe(outlier_df, use_container_width=True)
    
    # Encoding Information
    st.subheader("5️⃣ Informasi Encoding")
    
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    category_counts = df_processed[categorical_columns].nunique()
    
    encoding_info = pd.DataFrame({
        'Column': category_counts.index,
        'Unique Categories': category_counts.values
    })
    
    st.dataframe(encoding_info, use_container_width=True)
    
    # Store processed data
    st.session_state.df_processed = df_processed

# %% [markdown]
# # **7. Clustering Analysis Section**

elif menu == "Clustering Analysis":
    st.header("🔬 Clustering Analysis")
    
    if 'df_processed' not in st.session_state:
        st.error("❌ Silakan lakukan preprocessing terlebih dahulu!")
        st.stop()
    
    df_processed = st.session_state.df_processed
    
    # Clustering method selection
    clustering_method = st.selectbox(
        "Pilih Metode Clustering:",
        ["K-Means", "Gaussian Mixture Model (GMM)", "Agglomerative Hierarchical"]
    )
    
    st.subheader(f"📊 {clustering_method} Clustering")
    
    # Prepare data for clustering
    categorical_cols = ['SeriesDescription', 'GeoAreaName', 'Education level', 'Nature', 'Sex', 'Units']
    numerical_cols = ['TimePeriod', 'Value']
    
    # Filter columns that exist in the dataset
    categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
    numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
    
    if clustering_method == "K-Means":
        # K-Means implementation
        df_kmeans = df_processed.copy()
        
        # Label Encoding
        label_encoders_kmeans = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_kmeans[col] = le.fit_transform(df_kmeans[col])
            label_encoders_kmeans[col] = le
        
        # Standardization
        scaler_kmeans = StandardScaler()
        X_scaled_kmeans = scaler_kmeans.fit_transform(df_kmeans)
        
        # Determine optimal number of clusters
        st.write("**Menentukan Jumlah Cluster Optimal:**")
        
        with st.spinner("Menghitung Silhouette Score..."):
            scores_kmeans = []
            k_range = range(2, 8)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels_kmeans = kmeans.fit_predict(X_scaled_kmeans)
                score_kmeans = silhouette_score(X_scaled_kmeans, labels_kmeans)
                scores_kmeans.append(score_kmeans)
        
        # Plot silhouette scores
        fig_silhouette = px.line(
            x=list(k_range), 
            y=scores_kmeans,
            title="Silhouette Score untuk K-Means",
            labels={'x': 'Jumlah Cluster (k)', 'y': 'Silhouette Score'}
        )
        fig_silhouette.add_scatter(x=list(k_range), y=scores_kmeans, mode='markers', 
                                 marker=dict(size=8, color='red'))
        st.plotly_chart(fig_silhouette, use_container_width=True)
        
        # Select optimal k
        optimal_k = k_range[np.argmax(scores_kmeans)]
        st.success(f"✅ Jumlah cluster optimal: {optimal_k} (Silhouette Score: {max(scores_kmeans):.4f})")
        
        # Apply K-Means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels_kmeans = kmeans.fit_predict(X_scaled_kmeans)
        
        # PCA for visualization
        pca_kmeans = PCA(n_components=2)
        X_pca_kmeans = pca_kmeans.fit_transform(X_scaled_kmeans)
        
        # Visualization
        fig_cluster = px.scatter(
            x=X_pca_kmeans[:, 0], 
            y=X_pca_kmeans[:, 1],
            color=labels_kmeans.astype(str),
            title=f'Hasil K-Means Clustering (k={optimal_k})',
            labels={'x': 'PC 1', 'y': 'PC 2', 'color': 'Cluster'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Store results
        st.session_state.clustering_results = {
            'method': 'K-Means',
            'labels': labels_kmeans,
            'score': max(scores_kmeans),
            'n_clusters': optimal_k
        }
    
    elif clustering_method == "Gaussian Mixture Model (GMM)":
        # GMM implementation
        df_GMM = df_processed.copy()
        
        # One-Hot Encoding
        ohe_GMM = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_GMM = ohe_GMM.fit_transform(df_GMM[categorical_cols])
        
        # Combine with numerical data
        X_num_GMM = df_GMM[numerical_cols].values
        X_combined_GMM = np.hstack((X_cat_GMM, X_num_GMM))
        
        # Standardization
        scaler_GMM = StandardScaler()
        X_scaled_GMM = scaler_GMM.fit_transform(X_combined_GMM)
        
        # Find optimal number of components
        st.write("**Menentukan Jumlah Komponen Optimal:**")
        
        with st.spinner("Menghitung Silhouette Score untuk GMM..."):
            scores_gmm = []
            k_range = range(2, 8)
            
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=42)
                labels_GMM = gmm.fit_predict(X_scaled_GMM)
                score_GMM = silhouette_score(X_scaled_GMM, labels_GMM)
                scores_gmm.append(score_GMM)
        
        # Plot scores
        fig_gmm_scores = px.line(
            x=list(k_range), 
            y=scores_gmm,
            title="Silhouette Score untuk GMM",
            labels={'x': 'Jumlah Komponen', 'y': 'Silhouette Score'}
        )
        fig_gmm_scores.add_scatter(x=list(k_range), y=scores_gmm, mode='markers',
                                  marker=dict(size=8, color='green'))
        st.plotly_chart(fig_gmm_scores, use_container_width=True)
        
        # Apply GMM with optimal components
        optimal_components = k_range[np.argmax(scores_gmm)]
        st.success(f"✅ Jumlah komponen optimal: {optimal_components} (Silhouette Score: {max(scores_gmm):.4f})")
        
        gmm = GaussianMixture(n_components=optimal_components, random_state=42)
        labels_GMM = gmm.fit_predict(X_scaled_GMM)
        
        # PCA for visualization
        pca_GMM = PCA(n_components=2, random_state=42)
        X_pca_GMM = pca_GMM.fit_transform(X_scaled_GMM)
        
        # Visualization
        fig_gmm_cluster = px.scatter(
            x=X_pca_GMM[:, 0], 
            y=X_pca_GMM[:, 1],
            color=labels_GMM.astype(str),
            title=f'Hasil GMM Clustering (components={optimal_components})',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Cluster'}
        )
        st.plotly_chart(fig_gmm_cluster, use_container_width=True)
        
        # Store results
        st.session_state.clustering_results = {
            'method': 'GMM',
            'labels': labels_GMM,
            'score': max(scores_gmm),
            'n_clusters': optimal_components
        }
    
    elif clustering_method == "Agglomerative Hierarchical":
        # Agglomerative Clustering implementation
        df_copy = df_processed.copy()
        
        # One-Hot Encoding
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X_cat = ohe.fit_transform(df_copy[categorical_cols])
        
        # Combine with numerical data
        X_num = df_copy[numerical_cols].values
        X_all = np.hstack([X_cat, X_num])
        
        # Standardization
        scaler_Hierarchy = StandardScaler()
        X_scaled_Hierarchy = scaler_Hierarchy.fit_transform(X_all)
        
        # PCA for dimensionality reduction
        pca_Hierarchy = PCA(n_components=2)
        X_pca_Hierarchy = pca_Hierarchy.fit_transform(X_scaled_Hierarchy)
        
        # Try different numbers of clusters
        st.write("**Menentukan Jumlah Cluster Optimal:**")
        
        with st.spinner("Menghitung Silhouette Score untuk Agglomerative..."):
            scores_agglo = []
            k_range = range(2, 8)
            
            for k in k_range:
                agglo = AgglomerativeClustering(n_clusters=k)
                labels_agglo = agglo.fit_predict(X_pca_Hierarchy)
                score_agglo = silhouette_score(X_pca_Hierarchy, labels_agglo)
                scores_agglo.append(score_agglo)
        
        # Plot scores
        fig_agglo_scores = px.line(
            x=list(k_range), 
            y=scores_agglo,
            title="Silhouette Score untuk Agglomerative Clustering",
            labels={'x': 'Jumlah Cluster', 'y': 'Silhouette Score'}
        )
        fig_agglo_scores.add_scatter(x=list(k_range), y=scores_agglo, mode='markers',
                                   marker=dict(size=8, color='purple'))
        st.plotly_chart(fig_agglo_scores, use_container_width=True)
        
        # Apply Agglomerative with optimal clusters
        optimal_clusters = k_range[np.argmax(scores_agglo)]
        st.success(f"✅ Jumlah cluster optimal: {optimal_clusters} (Silhouette Score: {max(scores_agglo):.4f})")
        
        agglo_Hierarchy = AgglomerativeClustering(n_clusters=optimal_clusters)
        labels_agglo_Hierarchy = agglo_Hierarchy.fit_predict(X_pca_Hierarchy)
        
        # Visualization
        fig_agglo_cluster = px.scatter(
            x=X_pca_Hierarchy[:, 0], 
            y=X_pca_Hierarchy[:, 1],
            color=labels_agglo_Hierarchy.astype(str),
            title=f'Hasil Agglomerative Clustering (clusters={optimal_clusters})',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Cluster'}
        )
        st.plotly_chart(fig_agglo_cluster, use_container_width=True)
        
        # Store results and data for analysis
        st.session_state.clustering_results = {
            'method': 'Agglomerative',
            'labels': labels_agglo_Hierarchy,
            'score': max(scores_agglo),
            'n_clusters': optimal_clusters,
            'ohe': ohe,
            'X_cat': X_cat,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }

# %% [markdown]
# # **8. Results & Insights Section**

elif menu == "Results & Insights":
    st.header("📊 Results & Insights")
    
    if 'clustering_results' not in st.session_state:
        st.error("❌ Silakan lakukan clustering terlebih dahulu!")
        st.stop()
    
    if 'df_processed' not in st.session_state:
        st.error("❌ Data preprocessing belum dilakukan!")
        st.stop()
    
    results = st.session_state.clustering_results
    df_processed = st.session_state.df_processed
    
    st.subheader(f"🎯 Hasil {results['method']} Clustering")
    
    # Display clustering metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Metode Clustering", results['method'])
    
    with col2:
        st.metric("Jumlah Cluster", results['n_clusters'])
    
    with col3:
        st.metric("Silhouette Score", f"{results['score']:.4f}")
    
    # Prepare data for analysis
    if results['method'] == 'Agglomerative':
        # For Agglomerative, we need to reconstruct the original data
        ohe = results['ohe']
        X_cat = results['X_cat']
        categorical_cols = results['categorical_cols']
        numerical_cols = results['numerical_cols']
        
        # Inverse transform categorical data
        decoded_cats = ohe.inverse_transform(X_cat)
        df_decoded_cats = pd.DataFrame(decoded_cats, columns=categorical_cols)
        
        # Combine with numerical data
        df_original = pd.concat([df_decoded_cats, df_processed[numerical_cols].reset_index(drop=True)], axis=1)
        df_original['Cluster'] = results['labels'] + 1  # Add 1 to make clusters start from 1
    else:
        # For other methods, use the processed data directly
        df_original = df_processed.copy()
        df_original['Cluster'] = results['labels'] + 1
    
    # Cluster Analysis
    st.subheader("📈 Analisis Cluster")
    
    # 1. Cluster size distribution
    st.write("**1. Distribusi Ukuran Cluster:**")
    cluster_counts = df_original['Cluster'].value_counts().sort_index()
    
    fig_cluster_size = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        title="Distribusi Ukuran Cluster",
        labels={'x': 'Cluster', 'y': 'Jumlah Data Point'}
    )
    st.plotly_chart(fig_cluster_size, use_container_width=True)
    
    # 2. Value statistics per cluster
    if 'Value' in df_original.columns:
        st.write("**2. Statistik Nilai per Cluster:**")
        value_stats = df_original.groupby("Cluster")["Value"].describe()
        st.dataframe(value_stats, use_container_width=True)
        
        # Box plot for Value distribution
        fig_value_box = px.box(
            df_original, 
            x="Cluster", 
            y="Value",
            title="Distribusi Nilai per Cluster"
        )
        st.plotly_chart(fig_value_box, use_container_width=True)
        
        # Histogram for Value distribution
        fig_value_hist = px.histogram(
            df_original, 
            x="Value", 
            color="Cluster",
            title="Histogram Nilai per Cluster",
            barmode="overlay",
            opacity=0.7
        )
        st.plotly_chart(fig_value_hist, use_container_width=True)
    
    # 3. Categorical analysis
    st.write("**3. Analisis Kategorikal:**")
    
    categorical_cols_analysis = ['Education level', 'Sex', 'Nature']
    categorical_cols_analysis = [col for col in categorical_cols_analysis if col in df_original.columns]
    
    for col in categorical_cols_analysis:
        st.write(f"**{col} per Cluster:**")
        
        # Cross-tabulation
        crosstab = pd.crosstab(df_original["Cluster"], df_original[col], normalize='index') * 100
        st.dataframe(crosstab.round(2), use_container_width=True)
        
        # Stacked bar chart
        fig_cat = px.bar(
            crosstab.reset_index(), 
            x="Cluster",
            y=crosstab.columns.tolist(),
            title=f"Distribusi {col} per Cluster (%)",
            barmode="stack"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    # 4. Time period analysis
    if 'TimePeriod' in df_original.columns:
        st.write("**4. Analisis Periode Waktu:**")
        
        time_cluster = df_original.groupby(['Cluster', 'TimePeriod']).size().unstack(fill_value=0)
        
        fig_time = px.bar(
            time_cluster.reset_index(),
            x="Cluster",
            y=time_cluster.columns.tolist(),
            title="Distribusi Periode Waktu per Cluster",
            barmode="stack"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # 5. SeriesDescription analysis
    if 'SeriesDescription' in df_original.columns:
        st.write("**5. Top 3 SeriesDescription per Cluster:**")
        series_per_cluster = df_original.groupby("Cluster")["SeriesDescription"].apply(
            lambda x: x.value_counts().head(3)
        )
        
        for cluster in sorted(df_original['Cluster'].unique()):
            st.write(f"**Cluster {cluster}:**")
            cluster_series = series_per_cluster[cluster]
            for series, count in cluster_series.items():
                st.write(f"- {series}: {count} ({count/len(df_original[df_original['Cluster']==cluster])*100:.1f}%)")
    
    # 6. Cluster characteristics summary
    st.subheader("📋 Ringkasan Karakteristik Cluster")
    
    # Create summary table
    summary_data = []
    for cluster in sorted(df_original['Cluster'].unique()):
        cluster_data = df_original[df_original['Cluster'] == cluster]
        
        summary_row = {
            'Cluster': cluster,
            'Jumlah Data': len(cluster_data),
            'Persentase': f"{len(cluster_data)/len(df_original)*100:.1f}%"
        }
        
        # Add Value statistics if available
        if 'Value' in df_original.columns:
            summary_row['Rata-rata Value'] = f"{cluster_data['Value'].mean():.2f}"
            summary_row['Std Value'] = f"{cluster_data['Value'].std():.2f}"
        
        # Add dominant categories
        for col in ['Education level', 'Sex']:
            if col in df_original.columns:
                dominant = cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else 'N/A'
                summary_row[f'Dominan {col}'] = dominant
        
        # Add time period
        if 'TimePeriod' in df_original.columns:
            dominant_year = cluster_data['TimePeriod'].mode()[0] if len(cluster_data['TimePeriod'].mode()) > 0 else 'N/A'
            summary_row['Tahun Dominan'] = dominant_year
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # 7. Detailed cluster insights
    st.subheader("🔍 Insight Mendalam per Cluster")
    
    for cluster in sorted(df_original['Cluster'].unique()):
        with st.expander(f"📊 Cluster {cluster} - Detail Analysis"):
            cluster_data = df_original[df_original['Cluster'] == cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Statistik Dasar:**")
                st.write(f"- Jumlah data: {len(cluster_data):,}")
                st.write(f"- Persentase total: {len(cluster_data)/len(df_original)*100:.2f}%")
                
                if 'Value' in df_original.columns:
                    st.write(f"- Rata-rata Value: {cluster_data['Value'].mean():.2f}")
                    st.write(f"- Median Value: {cluster_data['Value'].median():.2f}")
                    st.write(f"- Range Value: {cluster_data['Value'].min():.2f} - {cluster_data['Value'].max():.2f}")
            
            with col2:
                st.write("**Karakteristik Dominan:**")
                
                # Education level
                if 'Education level' in cluster_data.columns:
                    edu_dist = cluster_data['Education level'].value_counts()
                    st.write(f"- Education level: {edu_dist.index[0]} ({edu_dist.iloc[0]/len(cluster_data)*100:.1f}%)")
                
                # Sex
                if 'Sex' in cluster_data.columns:
                    sex_dist = cluster_data['Sex'].value_counts()
                    st.write(f"- Sex: {sex_dist.index[0]} ({sex_dist.iloc[0]/len(cluster_data)*100:.1f}%)")
                
                # Time Period
                if 'TimePeriod' in cluster_data.columns:
                    time_dist = cluster_data['TimePeriod'].value_counts()
                    st.write(f"- Tahun: {time_dist.index[0]} ({time_dist.iloc[0]/len(cluster_data)*100:.1f}%)")
                
                # Nature
                if 'Nature' in cluster_data.columns:
                    nature_dist = cluster_data['Nature'].value_counts()
                    st.write(f"- Nature: {nature_dist.index[0]} ({nature_dist.iloc[0]/len(cluster_data)*100:.1f}%)")
    
    # 8. Geographic analysis if available
    if 'GeoAreaName' in df_original.columns:
        st.subheader("🌍 Analisis Geografis")
        
        # Top countries per cluster
        st.write("**Top 5 Negara per Cluster:**")
        
        for cluster in sorted(df_original['Cluster'].unique()):
            cluster_data = df_original[df_original['Cluster'] == cluster]
            top_countries = cluster_data['GeoAreaName'].value_counts().head(5)
            
            st.write(f"**Cluster {cluster}:**")
            for country, count in top_countries.items():
                percentage = count / len(cluster_data) * 100
                st.write(f"- {country}: {count} ({percentage:.1f}%)")
        
        # Geographic distribution visualization
        geo_cluster = df_original.groupby(['GeoAreaName', 'Cluster']).size().unstack(fill_value=0)
        
        # Show top 20 countries by total data points
        top_countries = df_original['GeoAreaName'].value_counts().head(20).index
        geo_cluster_top = geo_cluster.loc[top_countries]
        
        fig_geo = px.bar(
            geo_cluster_top.reset_index(),
            x="GeoAreaName",
            y=[f"{i}" for i in range(1, results['n_clusters'] + 1)],
            title="Distribusi Cluster per Negara (Top 20)",
            barmode="stack"
        )
        fig_geo.update_xaxes(tickangle=45)
        st.plotly_chart(fig_geo, use_container_width=True)
    
    # 9. Interpretation and recommendations
    st.subheader("💡 Interpretasi dan Rekomendasi")
    
    st.write("**Interpretasi Hasil Clustering:**")
    
    # Generate interpretations based on cluster characteristics
    interpretations = []
    
    for cluster in sorted(df_original['Cluster'].unique()):
        cluster_data = df_original[df_original['Cluster'] == cluster]
        
        interpretation = f"**Cluster {cluster}:**\n"
        
        # Value-based interpretation
        if 'Value' in df_original.columns:
            avg_value = cluster_data['Value'].mean()
            if avg_value > 70:
                interpretation += f"- Menunjukkan capaian pendidikan yang tinggi (rata-rata {avg_value:.1f}%)\n"
            elif avg_value > 40:
                interpretation += f"- Menunjukkan capaian pendidikan menengah (rata-rata {avg_value:.1f}%)\n"
            else:
                interpretation += f"- Menunjukkan capaian pendidikan yang rendah (rata-rata {avg_value:.1f}%)\n"
        
        # Education level interpretation
        if 'Education level' in cluster_data.columns:
            dominant_edu = cluster_data['Education level'].mode()[0]
            interpretation += f"- Didominasi oleh tingkat pendidikan: {dominant_edu}\n"
        
        # Time period interpretation
        if 'TimePeriod' in cluster_data.columns:
            dominant_year = cluster_data['TimePeriod'].mode()[0]
            interpretation += f"- Data terutama dari tahun: {dominant_year}\n"
        
        # Size interpretation
        cluster_size = len(cluster_data) / len(df_original) * 100
        if cluster_size > 40:
            interpretation += f"- Cluster terbesar ({cluster_size:.1f}% dari total data)\n"
        elif cluster_size < 20:
            interpretation += f"- Cluster minoritas ({cluster_size:.1f}% dari total data)\n"
        
        interpretations.append(interpretation)
    
    for interp in interpretations:
        st.write(interp)
    
    st.write("**Rekomendasi Kebijakan:**")
    
    recommendations = [
        "🎯 **Fokus pada cluster dengan capaian rendah:** Identifikasi dan berikan perhatian khusus pada cluster dengan nilai indikator pendidikan rendah",
        "📊 **Analisis temporal:** Perhatikan tren perubahan antar tahun untuk memahami perkembangan sistem pendidikan",
        "🌍 **Pendekatan geografis:** Pertimbangkan karakteristik regional dalam merancang kebijakan pendidikan",
        "⚖️ **Kesetaraan gender:** Pastikan program pendidikan dapat diakses secara setara oleh semua jenis kelamin",
        "📈 **Monitoring berkelanjutan:** Lakukan pemantauan rutin terhadap indikator-indikator kunci dalam setiap cluster"
    ]
    
    for rec in recommendations:
        st.write(rec)
    
    # 10. Export results
    st.subheader("💾 Export Hasil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download clustered data
        csv = df_original.to_csv(index=False)
        st.download_button(
            label="📥 Download Data Clustering (CSV)",
            data=csv,
            file_name="clustering_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download summary
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Ringkasan (CSV)",
            data=summary_csv,
            file_name="cluster_summary.csv",
            mime="text/csv"
        )
    
    # Performance metrics
    st.subheader("⚡ Performa Model")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            label="Silhouette Score",
            value=f"{results['score']:.4f}",
            help="Skor antara -1 hingga 1. Skor lebih tinggi menunjukkan clustering yang lebih baik."
        )
    
    with metrics_col2:
        # Calculate inertia if available
        if 'Value' in df_original.columns:
            cluster_centers = df_original.groupby('Cluster')['Value'].mean()
            within_cluster_variance = 0
            for cluster in df_original['Cluster'].unique():
                cluster_data = df_original[df_original['Cluster'] == cluster]['Value']
                within_cluster_variance += ((cluster_data - cluster_centers[cluster]) ** 2).sum()
            
            st.metric(
                label="Within-Cluster Variance",
                value=f"{within_cluster_variance:.2f}",
                help="Variance dalam cluster. Nilai lebih rendah menunjukkan cluster yang lebih kompak."
            )
    
    with metrics_col3:
        # Calculate between-cluster separation
        if 'Value' in df_original.columns:
            overall_mean = df_original['Value'].mean()
            between_cluster_variance = 0
            for cluster in df_original['Cluster'].unique():
                cluster_size = len(df_original[df_original['Cluster'] == cluster])
                cluster_mean = cluster_centers[cluster]
                between_cluster_variance += cluster_size * ((cluster_mean - overall_mean) ** 2)
            
            st.metric(
                label="Between-Cluster Variance",
                value=f"{between_cluster_variance:.2f}",
                help="Variance antar cluster. Nilai lebih tinggi menunjukkan cluster yang lebih terpisah."
            )

# Footer
st.markdown("---")
st.markdown("### 📌 Tentang Aplikasi")
st.markdown("""
**Clustering Goal 4 SDGs** adalah aplikasi analisis data yang menggunakan teknik machine learning 
untuk mengelompokkan data pendidikan berdasarkan indikator-indikator SDGs Goal 4. 

Aplikasi ini mendukung tiga metode clustering:
- **K-Means**: Clustering berbasis centroid
- **Gaussian Mixture Model (GMM)**: Clustering probabilistik
- **Agglomerative Hierarchical**: Clustering hierarkis

**Fitur Utama:**
- 📊 Eksplorasi data interaktif
- 🔧 Preprocessing otomatis
- 🎯 Multiple clustering algorithms
- 📈 Visualisasi hasil clustering
- 💡 Interpretasi dan rekomendasi kebijakan
- 💾 Export hasil analisis

---
*Dikembangkan untuk analisis SDGs Goal 4: Quality Education*
""")
