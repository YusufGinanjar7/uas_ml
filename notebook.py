import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Clustering SDGs Goal 4",
    page_icon="📊",
    layout="wide"
)

# Header aplikasi
st.title("🎓 Clustering Goal 4 (Pendidikan) - SDGs")
st.write("Hasil clustering dan visualisasi analisis SDGs menggunakan data Goal 4.")
st.markdown("---")

# Load dan preprocess data
@st.cache_data
def load_and_process_data():
    # Load data
    df = pd.read_csv('Goal4.csv')
    
    # Drop kolom yang tidak diperlukan
    columns_to_drop = [
        'TimeCoverage', 'UpperBound', 'LowerBound', 'GeoInfoUrl', 'FootNote',
        'BasePeriod', 'SeriesCode', 'GeoAreaCode', 'Time_Detail', 'Goal',
        'Target', 'Indicator', 'Source', 'Reporting Type', 'Quantile', 
        'Age', 'Type of skill', 'Location'
    ]
    
    # Hapus kolom yang ada
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Handle missing values
    df = df.dropna(subset=['Education level'])
    if 'Sex' in df.columns:
        df['Sex'].fillna(df['Sex'].mode()[0], inplace=True)
    
    return df

# Fungsi untuk melakukan clustering
@st.cache_data
def perform_clustering(df):
    # Define kolom kategorikal dan numerik
    categorical_cols = ['SeriesDescription', 'GeoAreaName', 'Education level', 'Nature', 'Sex', 'Units']
    numerical_cols = ['TimePeriod', 'Value']
    
    # Filter kolom yang benar-benar ada
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # One Hot Encoding untuk data kategorikal
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_cat = ohe.fit_transform(df[categorical_cols])
    
    # Ambil data numerik
    X_num = df[numerical_cols].values
    
    # Gabungkan data
    X_all = np.hstack([X_cat, X_num])
    
    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=3)
    labels = agglo.fit_predict(X_pca)
    
    # Hitung Silhouette Score
    silhouette_avg = silhouette_score(X_pca, labels)
    
    # Tambahkan hasil cluster ke dataframe
    df_result = df.copy()
    df_result['Cluster'] = labels + 1
    df_result['PCA_1'] = X_pca[:, 0]
    df_result['PCA_2'] = X_pca[:, 1]
    
    return df_result, silhouette_avg, pca

# Load data
try:
    df = load_and_process_data()
    st.success(f"✅ Data berhasil dimuat! Total: {len(df):,} baris")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Perform clustering
try:
    df_clustered, silhouette_avg, pca = perform_clustering(df)
    st.success(f"✅ Clustering berhasil! Silhouette Score: {silhouette_avg:.4f}")
except Exception as e:
    st.error(f"❌ Error dalam clustering: {str(e)}")
    st.stop()

# Sidebar untuk kontrol
st.sidebar.header("🎛️ Kontrol Visualisasi")
show_data = st.sidebar.checkbox("Tampilkan Data Hasil Clustering", value=True)
show_viz = st.sidebar.checkbox("Tampilkan Visualisasi", value=True)
show_analysis = st.sidebar.checkbox("Tampilkan Analisis Cluster", value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.metric("📊 Total Data", f"{len(df_clustered):,}")
    st.metric("🎯 Jumlah Cluster", "3")

with col2:
    st.metric("📈 Silhouette Score", f"{silhouette_avg:.4f}")
    st.metric("📋 Variabel PCA", f"{pca.explained_variance_ratio_.sum():.2%}")

st.markdown("---")

# Tampilkan data hasil clustering
if show_data:
    st.header("📋 Data Hasil Clustering")
    
    # Filter berdasarkan cluster
    cluster_filter = st.selectbox(
        "Pilih Cluster untuk ditampilkan:",
        ["Semua Cluster"] + [f"Cluster {i}" for i in sorted(df_clustered['Cluster'].unique())]
    )
    
    if cluster_filter == "Semua Cluster":
        data_to_show = df_clustered
    else:
        cluster_num = int(cluster_filter.split()[-1])
        data_to_show = df_clustered[df_clustered['Cluster'] == cluster_num]
    
    st.write(f"Menampilkan {len(data_to_show):,} baris data")
    st.dataframe(data_to_show, use_container_width=True)

# Visualisasi
if show_viz:
    st.header("📊 Visualisasi Hasil Clustering")
    
    # Tab untuk berbagai visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Scatter Plot", "📈 Distribusi Nilai", "📊 Boxplot", "🌍 Geografis"])
    
    with tab1:
        st.subheader("Scatter Plot - Hasil Clustering (PCA 2D)")
        
        # Plotly scatter plot
        fig_scatter = px.scatter(
            df_clustered, 
            x='PCA_1', 
            y='PCA_2', 
            color='Cluster',
            title=f'Agglomerative Clustering (PCA 2D) - Silhouette Score: {silhouette_avg:.4f}',
            labels={'PCA_1': 'PCA Component 1', 'PCA_2': 'PCA Component 2'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.subheader("Distribusi Nilai per Cluster")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]['Value']
            ax.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=30)
        
        ax.set_xlabel('Nilai Indikator')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi Nilai per Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Boxplot Nilai per Cluster")
        
        fig_box = px.box(
            df_clustered, 
            x='Cluster', 
            y='Value',
            title='Distribusi Nilai Tiap Cluster',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_box.update_layout(height=500)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        if 'GeoAreaName' in df_clustered.columns:
            st.subheader("Distribusi Geografis per Cluster")
            
            geo_cluster = df_clustered.groupby(['GeoAreaName', 'Cluster']).size().reset_index(name='Count')
            geo_dominant = geo_cluster.loc[geo_cluster.groupby('GeoAreaName')['Count'].idxmax()]
            
            fig_geo = px.bar(
                geo_dominant.head(20), 
                x='GeoAreaName', 
                y='Count',
                color='Cluster',
                title='Top 20 Negara/Wilayah berdasarkan Cluster Dominan',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_geo.update_xaxes(tickangle=45)
            fig_geo.update_layout(height=500)
            st.plotly_chart(fig_geo, use_container_width=True)

# Analisis Cluster
if show_analysis:
    st.header("🔍 Analisis Karakteristik Cluster")
    
    # Ringkasan statistik per cluster
    st.subheader("📊 Ringkasan Statistik per Cluster")
    
    cluster_summary = df_clustered.groupby('Cluster')['Value'].describe()
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Analisis komposisi kategorikal
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Education level' in df_clustered.columns:
            st.subheader("🎓 Komposisi Tingkat Pendidikan per Cluster")
            edu_crosstab = pd.crosstab(df_clustered['Cluster'], df_clustered['Education level'], normalize='index') * 100
            st.dataframe(edu_crosstab.round(2), use_container_width=True)
    
    with col2:
        if 'Sex' in df_clustered.columns:
            st.subheader("👥 Komposisi Jenis Kelamin per Cluster")
            sex_crosstab = pd.crosstab(df_clustered['Cluster'], df_clustered['Sex'], normalize='index') * 100
            st.dataframe(sex_crosstab.round(2), use_container_width=True)
    
    # Karakteristik dominan per cluster
    st.subheader("🏷️ Karakteristik Dominan per Cluster")
    
    # Buat ringkasan karakteristik
    characteristics = {}
    for cluster in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        
        char = {
            'Jumlah Data': len(cluster_data),
            'Rata-rata Value': cluster_data['Value'].mean(),
            'Median Value': cluster_data['Value'].median(),
            'Std Value': cluster_data['Value'].std()
        }
        
        # Tambahkan karakteristik kategorikal
        for col in ['Education level', 'Sex', 'Nature']:
            if col in cluster_data.columns:
                mode_val = cluster_data[col].mode()
                if len(mode_val) > 0:
                    char[f'{col} Dominan'] = mode_val.iloc[0]
        
        characteristics[f'Cluster {cluster}'] = char
    
    char_df = pd.DataFrame(characteristics).T
    st.dataframe(char_df, use_container_width=True)
    
    # Interpretasi Cluster
    st.subheader("💡 Interpretasi Karakteristik Cluster")
    
    st.markdown("""
    ### **Cluster 1: Pencapaian Tinggi**
    - **Rata-rata nilai indikator:** Tinggi (sekitar 77-80)
    - **Karakteristik:** Negara/wilayah dengan sistem pendidikan yang sudah matang dan efektif
    - **Tingkat pendidikan dominan:** Umumnya pendidikan dasar (PRIMAR)
    - **Analisis:** Mencerminkan daerah dengan kebijakan pendidikan universal yang berhasil
    
    ### **Cluster 2: Pencapaian Rendah** 
    - **Rata-rata nilai indikator:** Sangat rendah (mendekati 0-1)
    - **Karakteristik:** Wilayah dengan tantangan besar dalam pendidikan
    - **Tingkat pendidikan dominan:** Pendidikan menengah bawah (LOWSEC)
    - **Analisis:** Memerlukan intervensi khusus dan dukungan intensif
    
    ### **Cluster 3: Pencapaian Sedang**
    - **Rata-rata nilai indikator:** Sedang (sekitar 50-60)
    - **Karakteristik:** Wilayah dalam transisi peningkatan kualitas pendidikan
    - **Variasi:** Sebaran luas menunjukkan heterogenitas dalam pencapaian
    - **Analisis:** Berpotensi untuk peningkatan dengan strategi yang tepat
    """)

# Footer
st.markdown("---")
st.markdown("**📊 Dashboard Clustering SDGs Goal 4 - Analisis Pendidikan Global**")
st.markdown("*Menggunakan Agglomerative Hierarchical Clustering untuk mengidentifikasi pola pencapaian pendidikan*")

# Download hasil
if st.button("💾 Download Hasil Clustering"):
    csv = df_clustered.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="hasil_clustering_sdg_goal4.csv",
        mime="text/csv"
    )
