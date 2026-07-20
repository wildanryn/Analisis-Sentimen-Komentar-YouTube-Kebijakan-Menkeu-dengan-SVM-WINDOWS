import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="YouTube Sentiment App", layout="wide")

st.title("📊 YouTube Sentiment Analysis")

# ======================================================
# NLP SETUP (AMAN TANPA ERROR)
# ======================================================
@st.cache_resource
def setup_nlp():
    nltk.download('stopwords', quiet=True)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    return stemmer, stop_words

stemmer, stop_words = setup_nlp()

# ======================================================
# SCRAPING
# ======================================================
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None

@st.cache_data
def scrape_youtube(api_key, video_ids, max_results):
    youtube = build('youtube', 'v3', developerKey=api_key)
    data = []

    for vid in video_ids:
        next_page = None
        count = 0

        while count < max_results:
            res = youtube.commentThreads().list(
                part='snippet',
                videoId=vid,
                maxResults=100,
                pageToken=next_page,
                textFormat='plainText'
            ).execute()

            for item in res['items']:
                c = item['snippet']['topLevelComment']['snippet']
                data.append({'teks': c['textDisplay']})
                count += 1

            next_page = res.get('nextPageToken')
            if not next_page:
                break

    return pd.DataFrame(data)

# ======================================================
# PREPROCESS (TANPA NLTK TOKENIZER)
# ======================================================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ======================================================
# MENU
# ======================================================
menu = [
    "1. Scraping (Download CSV)",
    "2. Load CSV",
    "3. Preprocessing",
    "4. Training",
    "5. Prediksi",
    "6. Visualisasi"
]
choice = st.sidebar.radio("Navigasi", menu)

# ======================================================
# 1. SCRAPING
# ======================================================
if choice == "1. Scraping (Download CSV)":
    st.subheader("Scraping Komentar")

    api_key = st.text_input("API Key", type="password")
    links = st.text_area("Link YouTube (1 per baris)")
    max_c = st.slider("Jumlah komentar", 50, 5000, 100)

    if st.button("Mulai Scraping"):
        vids = [extract_video_id(l) for l in links.split("\n") if extract_video_id(l)]

        if not api_key:
            st.error("API Key wajib")
        elif not vids:
            st.error("Link tidak valid")
        else:
            df = scrape_youtube(api_key, vids, max_c)

            if not df.empty:
                st.success(f"Berhasil {len(df)} komentar")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "komentar.csv")

# ======================================================
# 2. LOAD CSV
# ======================================================
elif choice == "2. Load CSV":
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
                import pandas as pd
                import streamlit as st
                
                uploaded = st.file_uploader(
                    "Upload Dataset CSV",
                    type=["csv"]
                )
                
                if uploaded is not None:
                
                    # Daftar encoding yang akan dicoba
                    encodings = [
                        "utf-8",
                        "utf-8-sig",
                        "cp1252",
                        "latin1",
                        "ISO-8859-1"
                    ]
                
                    df = None
                
                    # Coba membaca dengan delimiter otomatis
                    for enc in encodings:
                        try:
                            uploaded.seek(0)
                            df = pd.read_csv(
                                uploaded,
                                encoding=enc,
                                sep=None,
                                engine="python"
                            )
                            st.success(f"✅ File berhasil dibaca menggunakan encoding: {enc}")
                            break
                
                        except Exception:
                            pass
                
                    # Jika gagal, coba delimiter koma
                    if df is None:
                        for enc in encodings:
                            try:
                                uploaded.seek(0)
                                df = pd.read_csv(
                                    uploaded,
                                    encoding=enc
                                )
                                st.success(f"✅ File berhasil dibaca menggunakan encoding: {enc}")
                                break
                
                            except Exception:
                                pass
                
                    # Jika masih gagal
                    if df is None:
                        st.error(
                            """
                            ❌ File CSV tidak dapat dibaca.
                
                            Kemungkinan penyebab:
                            - Encoding bukan UTF-8
                            - Separator salah
                            - File rusak
                            - File bukan CSV
                            """
                        )
                        st.stop()
                
                    # Tampilkan data
                    st.subheader("Preview Dataset")
                    st.dataframe(df)
                
                    # Informasi dataset
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        st.metric("Jumlah Baris", len(df))
                
                    with col2:
                        st.metric("Jumlah Kolom", len(df.columns))
                
                    with col3:
                        st.metric("Missing Value", int(df.isnull().sum().sum()))

        if 'teks' not in df.columns or 'label' not in df.columns:
            st.error("CSV harus ada kolom teks & label")
        else:
            st.session_state.df = df
            st.success("Data siap")
            st.dataframe(df.head())

# ======================================================
# 3. PREPROCESSING
# ======================================================
elif choice == "3. Preprocessing":

    st.subheader("🧹 Preprocessing Data")

    if 'df' not in st.session_state:
        st.warning("Upload data terlebih dahulu")
    else:

        df = st.session_state.df

        # ======================================================
        # FUNGSI PREPROCESS DETAIL
        # ======================================================
        def preprocess_detail(text):

            if pd.isna(text):
                return pd.Series([
                    "", "", "", "", ""
                ])

            # ======================
            # 1. CASE FOLDING
            # ======================
            case_folding = str(text).lower()

            # ======================
            # 2. CLEANING
            # ======================
            cleaning = re.sub(r'http\S+', '', case_folding)
            cleaning = re.sub(r'@\w+', '', cleaning)
            cleaning = re.sub(r'#\w+', '', cleaning)

            cleaning = re.sub(
                r'[^a-zA-Z\s]',
                ' ',
                cleaning
            )

            cleaning = re.sub(
                r'\s+',
                ' ',
                cleaning
            ).strip()

            # ======================
            # 3. TOKENIZING
            # ======================
            tokens = re.findall(
                r'\b\w+\b',
                cleaning
            )

            tokenizing = ", ".join(tokens)

            # ======================
            # 4. STOPWORD REMOVAL
            # ======================
            stop_tokens = [
                word
                for word in tokens
                if word not in stop_words
            ]

            stopword = ", ".join(stop_tokens)

            # ======================
            # 5. STEMMING
            # ======================
            stem_tokens = [
                stemmer.stem(word)
                for word in stop_tokens
            ]

            stemming = " ".join(stem_tokens)

            return pd.Series([
                case_folding,
                cleaning,
                tokenizing,
                stopword,
                stemming
            ])

        # ======================================================
        # PROSES PREPROCESSING
        # ======================================================
        if st.button("🚀 Proses Preprocessing"):

            with st.spinner("Sedang melakukan preprocessing..."):

                df[
                    [
                        'case_folding',
                        'cleaning',
                        'tokenizing',
                        'stopword',
                        'clean'
                    ]
                ] = df['teks'].apply(
                    preprocess_detail
                )

                st.session_state.df = df

            st.success("Preprocessing berhasil")

        # ======================================================
        # TAMPILKAN HASIL
        # ======================================================
        if 'clean' in df.columns:

            st.markdown("## 📋 Hasil Tahapan Preprocessing")

            jumlah_tampil = st.selectbox(
                "Jumlah data yang ditampilkan",
                [10, 25, 50, 100],
                index=0
            )

            # Membuat dataframe khusus tampilan
            df_tampil = df[
                [
                    'teks',
                    'case_folding',
                    'cleaning',
                    'tokenizing',
                    'stopword',
                    'clean'
                ]
            ].copy()

            # Mengubah nama kolom agar lebih rapi
            df_tampil.columns = [
                'Teks Asli',
                'Case Folding',
                'Cleaning',
                'Tokenizing',
                'Stopword Removal',
                'Stemming'
            ]

            # Tambahkan kolom sentimen jika tersedia
            if 'label' in df.columns:
                df_tampil['label'] = df['label']

            st.dataframe(
                df_tampil.head(jumlah_tampil),
                use_container_width=True,
                height=500
            )

            # ======================================================
            # STATISTIK
            # ======================================================
            st.markdown("## 📊 Statistik Preprocessing")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Jumlah Data",
                    len(df)
                )

            with col2:
                rata_sebelum = (
                    df['teks']
                    .astype(str)
                    .apply(lambda x: len(x.split()))
                    .mean()
                )

                st.metric(
                    "Rata-rata Kata Sebelum",
                    f"{rata_sebelum:.2f}"
                )

            with col3:
                rata_sesudah = (
                    df['clean']
                    .astype(str)
                    .apply(lambda x: len(x.split()))
                    .mean()
                )

                st.metric(
                    "Rata-rata Kata Sesudah",
                    f"{rata_sesudah:.2f}"
                )


# ======================================================
# 4. TRAINING
# ======================================================
elif choice == "4. Training":

    st.subheader("🤖 Training Model SVM")

    if 'df' not in st.session_state:
        st.warning("Upload data terlebih dahulu")
    elif 'clean' not in st.session_state.df.columns:
        st.warning("Lakukan preprocessing terlebih dahulu")
    else:

        df = st.session_state.df

        # ======================================================
        # HAPUS DATA KOSONG
        # ======================================================
        before = len(df)

        df = df.dropna(subset=['clean', 'label'])

        df = df[
            (df['clean'].astype(str).str.strip() != '') &
            (df['label'].astype(str).str.strip() != '')
        ]

        after = len(df)

        st.info(f"{before - after} data kosong dihapus")

        # ======================================================
        # VALIDASI DATA
        # ======================================================
        if len(df) == 0:
            st.error("Data kosong setelah cleaning")
            st.stop()

        # ======================================================
        # PENGATURAN SPLIT DATA
        # ======================================================
        st.markdown("## ⚙️ Pengaturan Data")

        test_size = st.slider(
            "Persentase Data Testing",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )

        train_size = 100 - test_size

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Data Training",
                f"{train_size}%"
            )

        with col2:
            st.metric(
                "Data Testing",
                f"{test_size}%"
            )

        # ======================================================
        # INFO TOTAL DATA
        # ======================================================
        total_data = len(df)

        total_train = int((train_size / 100) * total_data)
        total_test = int((test_size / 100) * total_data)

        st.info(
            f"""
            Total Data : {total_data}

            • Training : {total_train}
            • Testing : {total_test}
            """
        )

        # ======================================================
        # BUTTON TRAIN
        # ======================================================
        if st.button("🚀 Train Model"):

            with st.spinner("Sedang melatih model..."):

                # ======================================================
                # FITUR & LABEL
                # ======================================================
                X = df['clean']
                X_text = df['teks']
                y = df['label']
                # ======================================================
                # TF-IDF
                # ======================================================
                vectorizer = TfidfVectorizer(
                    max_features=3000
                )

                X_vector = vectorizer.fit_transform(X)

                # ======================================================
                # SPLIT DATA
                # ======================================================
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vector,
                    y,
                    test_size=test_size / 100,
                    random_state=42,
                    stratify=y
                )

                # ======================================================
                # MODEL SVM
                # ======================================================
                model = SVC(
                    kernel='linear',
                    random_state=42
                )

                model.fit(X_train, y_train)

                # ======================================================
                # PREDIKSI
                # ======================================================
                y_pred = model.predict(X_test)
                hasil_prediksi = pd.DataFrame({
                    'Aktual': y_test.values,
                    'Prediksi': y_pred
                })

                st.session_state.hasil_prediksi = hasil_prediksi

                X_test_text = X_text.loc[y_test.index]

                st.session_state.test_text = X_test_text.reset_index(drop=True)
                # ======================================================
                # AKURASI
                # ======================================================
                acc = accuracy_score(y_test, y_pred)

                st.success("Training selesai")

                st.metric(
                    "Akurasi Model",
                    f"{acc * 100:.2f}%"
                )

                # ======================================================
                # SIMPAN MODEL
                # ======================================================
                st.session_state.model = model
                st.session_state.vec = vectorizer

                # ======================================================
                # HASIL SPLIT
                # ======================================================
                st.markdown("## 📊 Pembagian Data")

                split_df = pd.DataFrame({
                    'Jenis': ['Training', 'Testing'],
                    'Jumlah': [len(y_train), len(y_test)],
                    'Persentase': [
                        f'{train_size}%',
                        f'{test_size}%'
                    ]
                })

                st.dataframe(
                    split_df,
                    use_container_width=True
                )

                # ======================================================
                # CONFUSION MATRIX
                # ======================================================
                st.markdown("## 🔥 Confusion Matrix")

                labels = sorted(y.unique())

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=labels
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=labels,
                    columns=labels
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                # ======================================================
                # VISUAL CONFUSION MATRIX
                # ======================================================
                fig, ax = plt.subplots(figsize=(6, 4))

                im = ax.imshow(cm)

                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))

                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

                plt.xlabel("Prediksi")
                plt.ylabel("Aktual")
                plt.title("Confusion Matrix")

                # ======================================================
                # TAMPILKAN ANGKA
                # ======================================================
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            color="black"
                        )

                st.pyplot(fig)

                # ======================================================
                # CLASSIFICATION REPORT
                # ======================================================
                st.markdown("## 📋 Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(
                    report_df,
                    use_container_width=True
                )

# ======================================================
# 5. PREDIKSI
# ======================================================
elif choice == "5. Prediksi":
    if 'model' not in st.session_state:
        st.warning("Model belum ada")
    else:
        text = st.text_area("Komentar")

        if st.button("Prediksi"):
            clean = preprocess(text)
            vec = st.session_state.vec.transform([clean])
            pred = st.session_state.model.predict(vec)

            st.success(f"Hasil: {pred[0]}")

# ======================================================
# 6. VISUALISASI
# ======================================================
elif choice == "6. Visualisasi":

    st.subheader("📊 Visualisasi Hasil")

    if 'hasil_prediksi' not in st.session_state:
        st.warning("Lakukan training terlebih dahulu")
    else:

        hasil = st.session_state.hasil_prediksi
        
        # ======================================
        # PIE CHART LABEL MANUAL
        # ======================================

        st.markdown("## 🥧 Distribusi Hasil Pelabelan")

        if 'df' in st.session_state:

            df_label = st.session_state.df

            if 'label' in df_label.columns:

                label_counts = df_label['label'].value_counts()

                fig_label, ax_label = plt.subplots(figsize=(6, 6))

                ax_label.pie(
                    label_counts,
                    labels=label_counts.index,
                    autopct='%1.1f%%'
                )

                ax_label.set_title("Distribusi Hasil Pelabelan")

                ax_label.axis('equal')

                st.pyplot(fig_label)
     
        distribusi_label = label_counts.reset_index()
        distribusi_label.columns = ['Label', 'Jumlah']

        st.dataframe(
            distribusi_label,
            use_container_width=True
        )
        # ===============================
        # PIE CHART HASIL PREDIKSI
        # ===============================
        st.markdown("## 🥧 Distribusi Prediksi SVM")
        counts = hasil['Prediksi'].value_counts()

        fig1, ax1 = plt.subplots(figsize=(6,6))

        ax1.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%'
        )

        ax1.axis('equal')

        st.pyplot(fig1)

        # ===============================
        # TABEL HASIL PREDIKSI
        # ===============================
        st.markdown("## 📋 Hasil Prediksi")

        st.dataframe(
            hasil,
            use_container_width=True
        )

        # ===============================
        # WORDCLOUD BERDASARKAN PREDIKSI
        # ===============================
        if 'test_text' in st.session_state:

            teks_uji = st.session_state.test_text

            hasil_wc = pd.DataFrame({
                'teks': teks_uji,
                'prediksi': hasil['Prediksi']
            })

            st.markdown("## ☁️ Wordcloud Berdasarkan Prediksi SVM")

            labels = hasil_wc['prediksi'].unique()

            cols = st.columns(len(labels))

            for i, label in enumerate(labels):

                text_data = " ".join(
                    hasil_wc[
                        hasil_wc['prediksi'] == label
                    ]['teks'].astype(str)
                )

                with cols[i]:

                    st.write(f"### {label}")

                    if text_data.strip():

                        wc = WordCloud(
                            width=600,
                            height=300,
                            background_color='white'
                        ).generate(text_data)

                        fig, ax = plt.subplots()

                        ax.imshow(wc)

                        ax.axis('off')

                        st.pyplot(fig)

                    else:
                        st.write("Tidak ada data")
