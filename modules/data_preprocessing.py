import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import re
import emoji
from io import BytesIO
import warnings
import time
warnings.filterwarnings('ignore')
import numpy as np
import io
from datetime import datetime
import ast
from tqdm import tqdm
tqdm.pandas()
@st.cache_resource
def set_langdetect_seed():
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
set_langdetect_seed()

# Import libraries untuk preprocessing
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect library tidak tersedia. Menggunakan deteksi script saja.")

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    st.warning("⚠️ Library langdetect atau deep_translator tidak terinstall. Fitur deteksi bahasa dan translate tidak tersedia.")

def show():
    st.title("Data & Preprocessing")
    st.markdown("---")
    
    # Tab untuk berbagai bagian
    tab1, tab2, tab3, tab4 = st.tabs(["Data Integration & Overview", "Preprocessing Steps", "Upload & Process Data", "Visualisasi"])
    
    with tab1:
        show_data_overview()
    
    with tab2:
        show_preprocessing_steps()
    
    with tab3:
        show_upload_data()
    
    with tab4:
        show_visualizations()

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

def show_data_overview():
    """Menampilkan proses preprocessing data lengkap"""
    st.header("Data Integration & Initial Assessment")
    
    st.markdown("""
    ### Tahapan Preprocessing Data
    Proses ini akan melakukan:
    1. **Merge Data**: Menggabungkan multiple file data (CSV/Excel)
    2. **Rename Column**: Mengubah nama kolom `created_at` menjadi `tanggal`
    3. **Format Tanggal**: Konversi ke datetime format
    4. **Sort Data**: Mengurutkan berdasarkan tanggal (ascending)
    5. **Hapus Duplikat**: Menghapus data duplikat
    6. **Download**: Menyimpan data yang telah diproses
    """)

    st.markdown("""
    ### Petunjuk Upload Data:
    
    **Format File**: Excel (.xlsx) atau CSV (.csv)
    
    **Kolom yang Diperlukan**:
    - `id` atau `id_str`: ID unik tweet
    - `tanggal` atau `created_at`: Timestamp tweet
    - `tweet` atau `full_text`: Konten tweet
    - `like` atau `favorite_count`: Jumlah like
    - `quote` atau `quote_count`: Jumlah quote
    - `reply` atau `reply_count`: Jumlah reply
    - `retweet` atau `retweet_count`: Jumlah retweet
    - `username`: Username akun
    - `followers`: Jumlah followers
    - `following`: Jumlah following
    - `verified_status`: Status verifikasi
    """)
    
    # File uploader untuk multiple files
    st.subheader("Step 1: Upload File Data")
    uploaded_files = st.file_uploader(
        "Upload satu atau lebih file (CSV/Excel)", 
        type=['xlsx', 'csv', 'xls'],
        accept_multiple_files=True,
        key='preprocessing_upload'
    )
    
    if uploaded_files:
        if st.button("Mulai Preprocessing", type="primary"):
            with st.spinner("Memproses data..."):
                # ===========================================
                # STEP 1: MERGE DATA
                # ===========================================
                st.subheader("Step 1: Merge Data dari Multiple Files")
                
                df_list = []
                merge_info = []
                
                for file in uploaded_files:
                    try:
                        if file.name.endswith('.csv'):
                            temp_df = pd.read_csv(file)
                        else:  # xlsx atau xls
                            temp_df = pd.read_excel(file)
                        
                        df_list.append(temp_df)
                        merge_info.append({
                            'Filename': file.name,
                            'Rows': len(temp_df),
                            'Columns': len(temp_df.columns)
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Gagal membaca {file.name}: {str(e)}")
                        continue
                
                if len(df_list) == 0:
                    st.error("❌ Tidak ada file yang berhasil dibaca!")
                    return
                
                # Gabungkan semua DataFrame
                df = pd.concat(df_list, ignore_index=True)
                
                # Tampilkan info merge
                merge_df = pd.DataFrame(merge_info)
                st.success(f"✅ Berhasil menggabungkan {len(df_list)} file")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", len(df_list))
                with col2:
                    st.metric("Total Rows", len(df))
                with col3:
                    st.metric("Total Columns", len(df.columns))
                
                with st.expander("Detail Merge per File"):
                    st.dataframe(merge_df, use_container_width=True)
                
                st.divider()
                
                # ===========================================
                # STEP 2: RENAME COLUMN
                # ===========================================
                st.subheader("Step 2: Rename Column 'created_at' → 'tanggal'")
                
                if 'created_at' in df.columns:
                    df = df.rename(columns={'created_at': 'tanggal'})
                    st.success("✅ Kolom berhasil direname: `created_at` → `tanggal`")
                elif 'tanggal' in df.columns:
                    st.info("ℹ️ Kolom `tanggal` sudah ada, skip rename")
                else:
                    st.warning("⚠️ Kolom `created_at` atau `tanggal` tidak ditemukan!")
                
                st.divider()
                
                # ===========================================
                # STEP 3: FORMAT TANGGAL
                # ===========================================
                st.subheader("Step 3: Konversi Format Tanggal ke Datetime")
                
                if 'tanggal' in df.columns:
                    try:
                        df['tanggal'] = pd.to_datetime(
                            df['tanggal'],
                            errors='coerce',
                            utc=True
                        ).dt.tz_localize(None)
                        
                        # if hasattr(df['tanggal'].dt, "tz"):
                        #     df['tanggal'] = df['tanggal'].dt.tz_localize(None)

                        # df['tanggal'] = pd.to_datetime(
                        #     df['tanggal'].astype(str),
                        #     errors='coerce',
                        #     utc=True
                        # ).dt.tz_localize(None)
                        
                        st.success("✅ Kolom `tanggal` berhasil dikonversi ke datetime")
                        
                        # Info tanggal
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Tanggal Terawal", df['tanggal'].min().strftime('%Y-%m-%d %H:%M:%S'))
                        with col2:
                            st.metric("Tanggal Terakhir", df['tanggal'].max().strftime('%Y-%m-%d %H:%M:%S'))
                    
                    except Exception as e:
                        st.error(f"❌ Error saat konversi tanggal: {str(e)}")
                else:
                    st.warning("⚠️ Kolom `tanggal` tidak ditemukan, skip konversi")
                
                st.divider()
                
                # ===========================================
                # STEP 4: SORT BY TANGGAL
                # ===========================================
                st.subheader("Step 4: Mengurutkan Data Berdasarkan Tanggal")
                
                if 'tanggal' in df.columns:
                    df = df.sort_values(by='tanggal', ascending=True)
                    df = df.reset_index(drop=True)
                    st.success("✅ Data berhasil diurutkan berdasarkan tanggal (ascending)")
                    
                    # Preview data terurut
                    with st.expander("Preview 5 Data Pertama (Tanggal Terawal)"):
                        st.dataframe(df.head(), use_container_width=True)
                else:
                    st.warning("⚠️ Kolom `tanggal` tidak ada, skip sorting")
                
                st.divider()
                
                # # ===========================================
                # # STEP 5: DETEKSI SPAM/BUZZER
                # # ===========================================
                # st.subheader("Step 5: Pengecekan Indikasi Spam/Buzzer")
                
                # # Analisis Content Diversity
                # spam_results = analyze_spam_buzzer(df)
                
                # # Tampilkan hasil analisis
                # st.markdown("### Hasil Analisis Spam/Buzzer (6 Analisis)")
                
                # # Analisis 1: Content Diversity
                # with st.expander("Analisis 1: Keberagaman Konten", expanded=True):
                #     col1, col2, col3 = st.columns(3)
                #     with col1:
                #         st.metric("Total Tweets", spam_results['content_diversity']['total_tweets'])
                #     with col2:
                #         st.metric("Unique Tweets", spam_results['content_diversity']['unique_tweets'])
                #     with col3:
                #         st.metric("Duplicate Tweets", spam_results['content_diversity']['duplicate_tweets'])
                    
                #     diversity = spam_results['content_diversity']['diversity_ratio']
                #     if diversity > 70:
                #         st.success(f"✅ Diversity Ratio: {diversity:.2f}% - Konten BERAGAM (BUKAN spam/buzzer)")
                #     elif diversity > 50:
                #         st.warning(f"⚠️ Diversity Ratio: {diversity:.2f}% - Ada beberapa duplikasi")
                #     else:
                #         st.error(f"❌ Diversity Ratio: {diversity:.2f}% - Konten REPETITIF (potensi spam/buzzer)")
                    
                #     st.markdown("**Top 5 Tweets yang Paling Sering Muncul:**")
                #     for idx, item in enumerate(spam_results['content_diversity']['top_duplicates'][:5], 1):
                #         st.markdown(f"{idx}. `{item['count']}x` - {item['text'][:80]}...")
                
                # # Analisis 2: User Behavior
                # with st.expander("Analisis 2: User Behavior"):
                #     col1, col2, col3 = st.columns(3)
                #     with col1:
                #         st.metric("Total Users", spam_results['user_behavior']['total_users'])
                #     with col2:
                #         st.metric("Avg User Diversity", f"{spam_results['user_behavior']['avg_user_diversity']:.2f}%")
                #     with col3:
                #         st.metric("Spam Users", f"{spam_results['user_behavior']['spam_users_count']} ({spam_results['user_behavior']['spam_user_pct']:.2f}%)")
                    
                #     if spam_results['user_behavior']['is_organic']:
                #         st.success("✅ User behavior SEHAT - Mayoritas posting konten beragam")
                #     else:
                #         st.warning("⚠️ Ada indikasi user behavior yang mencurigakan")
                    
                #     st.markdown("**Top 5 Users dengan Konten Paling Repetitif:**")
                #     for idx, user in enumerate(spam_results['user_behavior']['top_spam_users'][:5], 1):
                #         st.markdown(
                #             f"{idx}. **{user['username']}** - "
                #             f"Total: {user['total']}, Unique: {user['unique']}, "
                #             f"Diversity: {user['diversity']:.1f}%"
                #         )
                
                # # Analisis 3: Temporal Pattern
                # with st.expander("Analisis 3: Pola Temporal"):
                #     if 'temporal_pattern' in spam_results and spam_results['temporal_pattern']:
                #         col1, col2 = st.columns(2)

                #         with col1:
                #             st.metric(
                #                 "Coefficient of Variation",
                #                 f"{spam_results['temporal_pattern']['cv']:.2f}%"
                #             )

                #         with col2:
                #             if spam_results['temporal_pattern']['is_natural']:
                #                 st.success("✅ Pola Cenderung NATURAL")
                #             else:
                #                 st.error("❌ Pola TIDAK NATURAL")

                #         st.markdown("**Interpretasi:**")
                #         st.markdown(spam_results['temporal_pattern']['interpretation'])

                #         st.markdown("**Distribusi Tweets per Jam:**")
                #         hourly = spam_results['temporal_pattern']['hourly_dist']

                #         # Pastikan data diurutkan berdasarkan jam (0-23)
                #         if hasattr(hourly, 'reindex'):
                #             # Jika pandas Series
                #             hourly = hourly.reindex(range(24), fill_value=0)
                #             hours = hourly.index.tolist()
                #             values = hourly.values.tolist()
                #         else:
                #             # Jika dict
                #             hours = list(range(24))
                #             values = [hourly.get(h, 0) for h in hours]

                #         fig, ax = plt.subplots(figsize=(10, 5))
                #         bars = ax.bar(hours, values)
                #         ax.bar_label(bars, labels=values, padding=3, fontsize=8)
                #         ax.set_title("Distribusi Tweet per Jam")
                #         ax.set_xlabel("Jam")
                #         ax.set_ylabel("Jumlah Tweet")
                #         ax.set_xticks(range(24))
                #         ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha='right')
                #         ax.set_ylim(0, max(values) * 1.12)
                #         st.pyplot(fig)
                #         # st.bar_chart(spam_results['temporal_pattern']['hourly_dist'])
                #         # fig, ax = plt.subplots()
                #         # ax.bar(range(24), spam_results['temporal_pattern']['hourly_dist'])
                #         # st.pyplot(fig)
                #     else:
                #         st.info("ℹ️ Kolom timestamp tidak tersedia untuk analisis temporal")

                
                # # Analisis 4: Text Characteristics
                # with st.expander("Analisis 4: Karakteristik Teks"):
                #     col1, col2, col3 = st.columns(3)
                #     with col1:
                #         st.metric("Avg Length", f"{spam_results['text_characteristics']['avg_length']:.0f} char")
                #     with col2:
                #         st.metric("URL %", f"{spam_results['text_characteristics']['url_pct']:.1f}%")
                #     with col3:
                #         st.metric("Hashtag %", f"{spam_results['text_characteristics']['hashtag_pct']:.1f}%")
                    
                #     spam_score = spam_results['text_characteristics']['spam_score']
                #     if spam_score == 0:
                #         st.success("✅ Karakteristik teks NORMAL")
                #         st.caption("Tidak ada indikator spam berdasarkan karakteristik teks")
                #     else:
                #         st.warning(f"⚠️ Ada {spam_score} indikator spam dari karakteristik teks")
                #         for reason in spam_results['text_characteristics']['spam_reasons']:
                #             st.markdown(f"- {reason}")      
                
                # # Analisis 5: Hashtag Diversity
                # with st.expander("Analisis 5: Keberagaman Hashtag"):
                #     if spam_results['hashtag_diversity']['total_hashtags'] > 0:
                #         col1, col2, col3 = st.columns(3)
                #         with col1:
                #             st.metric("Total Hashtags", spam_results['hashtag_diversity']['total_hashtags'])
                #         with col2:
                #             st.metric("Unique Hashtags", spam_results['hashtag_diversity']['unique_hashtags'])
                #         with col3:
                #             st.metric("Diversity", f"{spam_results['hashtag_diversity']['diversity']:.1f}%")
                        
                #         if spam_results['hashtag_diversity']['is_diverse']:
                #             st.success("✅ Hashtag BERAGAM - Diskusi organik")
                #         else:
                #             st.warning("⚠️ Hashtag kurang beragam - Potensi campaign")
                        
                #         st.markdown("**Top 10 Hashtag:**")
                #         for idx, item in enumerate(spam_results['hashtag_diversity']['top_hashtags'][:10], 1):
                #             st.markdown(f"{idx}. #{item['tag']}: {item['count']}x ({item['pct']:.1f}%)")
                #     else:
                #         st.info("ℹ️ Tidak ada hashtag ditemukan dalam dataset")
                
                # # Analisis 6: Sentiment Variation
                # with st.expander("Analisis 6: Variasi Sentimen"):
                #     col1, col2 = st.columns(2)
                #     with col1:
                #         st.metric("Diversity Score", f"{spam_results['sentiment']['diversity_score']:.1f}%")
                #     with col2:
                #         if spam_results['sentiment']['is_diverse']:
                #             st.success("✅ Sentimen BERAGAM")
                #         else:
                #             st.warning("⚠️ Sentimen SERAGAM")
                    
                #     st.markdown("**Distribusi Sentimen:**")
                #     sentiment_data = spam_results['sentiment']['distribution']
                #     for sent, count in sentiment_data.items():
                #         pct = spam_results['sentiment']['distribution_pct'][sent]
                #         st.markdown(f"- **{sent.upper()}**: {count} ({pct:.1f}%)")
                
                # # Final Report
                # st.markdown("---")
                # st.markdown("### Laporan Akhir")
                
                # final = spam_results['final_report']
                
                # col1, col2, col3 = st.columns(3)
                # with col1:
                #     st.metric("Total Checks", final['total_checks'])
                # with col2:
                #     st.metric("Passed Checks", final['passed_checks'])
                # with col3:
                #     st.metric("Pass Rate", f"{final['pass_rate']:.1f}%")
                
                # if final['pass_rate'] >= 70:
                #     st.success("✅ KESIMPULAN: DATA BERSIH - Dataset menunjukkan karakteristik ORGANIK")
                # elif final['pass_rate'] >= 50:
                #     st.warning("⚠️ KESIMPULAN: DATA CUKUP BERSIH - Ada beberapa indikator yang perlu diperhatikan")
                # else:
                #     st.error("❌ KESIMPULAN: DATA BERMASALAH - Banyak indikator spam/buzzer activity")
                
                # st.divider()
                
                # ===========================================
                # STEP 6: HAPUS DUPLIKAT
                # ===========================================
                st.subheader("Step 5: Pengecekan dan Penghapusan Data Duplikat")
                
                total_duplicates = df.duplicated().sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Data Sebelum", len(df))
                    st.metric("Duplikat Terdeteksi", total_duplicates)
                with col2:
                    if total_duplicates > 0:
                        st.warning(f"⚠️ Ditemukan {total_duplicates} data duplikat")
                        df = df.drop_duplicates().reset_index(drop=True)
                        st.success(f"✅ Duplikat berhasil dihapus!")
                    else:
                        st.success("✅ Tidak ada duplikat")
                    
                    st.metric("Total Data Setelah", len(df))
                
                st.divider()
                
                # ===========================================
                # STEP 7: PREVIEW & DOWNLOAD
                # ===========================================
                st.subheader("Step 6: Preview & Download Data Hasil Preprocessing")
                
                # Info final
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows (Final)", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                
                # Preview data
                st.markdown("**Preview Data (10 baris pertama):**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data info
                with st.expander("Info Kolom & Tipe Data"):
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                
                # Download buttons
                st.markdown("### Download Data Hasil Preprocessing")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download Excel
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Excel (.xlsx)",
                        data=excel_buffer,
                        file_name=f"data_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    # Download CSV
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download CSV (.csv)",
                        data=csv_data,
                        file_name=f"data_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Simpan ke session state
                st.session_state['df_preprocessed'] = df
                st.success("✅ Data preprocessing selesai! Data tersimpan di session state.")
    
    else:
        st.info("Upload minimal 1 file data untuk memulai preprocessing")


# def analyze_spam_buzzer(df):
#     """
#     Fungsi untuk menganalisis indikasi spam/buzzer dengan 6 analisis lengkap
#     """
#     import re
#     from collections import Counter
    
#     results = {}
    
#     # Tentukan kolom yang digunakan
#     text_col = 'full_text' if 'full_text' in df.columns else 'tweet'
#     user_col = 'username' if 'username' in df.columns else 'user'
#     time_col = 'tanggal' if 'tanggal' in df.columns else 'created_at'
    
#     # ========================================
#     # ANALISIS 1: CONTENT DIVERSITY
#     # ========================================
#     total_tweets = len(df)
#     unique_tweets = df[text_col].nunique()
#     duplicate_tweets = total_tweets - unique_tweets
#     diversity_ratio = (unique_tweets / total_tweets) * 100
    
#     top_dup = df[text_col].value_counts().head(10)
#     top_duplicates = [
#         {'text': text, 'count': count} 
#         for text, count in top_dup.items()
#     ]
    
#     results['content_diversity'] = {
#         'total_tweets': total_tweets,
#         'unique_tweets': unique_tweets,
#         'duplicate_tweets': duplicate_tweets,
#         'diversity_ratio': diversity_ratio,
#         'top_duplicates': top_duplicates,
#         'is_diverse': diversity_ratio > 70,
#         'interpretation': 'Konten beragam' if diversity_ratio > 70 else 'Konten repetitif'
#     }
    
#     # ========================================
#     # ANALISIS 2: USER BEHAVIOR
#     # ========================================
#     user_diversity = {}
#     spam_users = []
    
#     for user in df[user_col].unique():
#         user_tweets = df[df[user_col] == user][text_col]
#         total = len(user_tweets)
#         unique = user_tweets.nunique()
#         diversity = (unique / total * 100) if total > 0 else 0
        
#         user_diversity[user] = {
#             'total_tweets': total,
#             'unique_tweets': unique,
#             'diversity': diversity
#         }
        
#         if diversity < 50 and total >= 5:
#             spam_users.append(user)
    
#     total_users = len(user_diversity)
#     spam_user_count = len(spam_users)
#     spam_user_pct = (spam_user_count / total_users * 100) if total_users > 0 else 0
#     avg_diversity = np.mean([u['diversity'] for u in user_diversity.values()])
    
#     sorted_users = sorted(user_diversity.items(), key=lambda x: x[1]['diversity'])
#     top_spam_users = [
#         {
#             'username': user,
#             'total': stats['total_tweets'],
#             'unique': stats['unique_tweets'],
#             'diversity': stats['diversity']
#         }
#         for user, stats in sorted_users[:5]
#     ]
    
#     is_organic = spam_user_pct < 10 and avg_diversity > 70
    
#     results['user_behavior'] = {
#         'total_users': total_users,
#         'avg_user_diversity': avg_diversity,
#         'spam_users_count': spam_user_count,
#         'spam_user_pct': spam_user_pct,
#         'top_spam_users': top_spam_users,
#         'is_organic': is_organic,
#         'interpretation': 'Organik' if is_organic else 'Ada indikasi spam'
#     }
    
#     # ========================================
#     # ANALISIS 3: TEMPORAL PATTERN
#     # ========================================
#     if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
#         try:
#             # Ambil jam TANPA mengubah datetime
#             hour_series = df[time_col].dt.hour

#             # Distribusi jam 0–23 (WAJIB lengkap)
#             hourly_dist = (
#                 hour_series
#                 .value_counts()
#                 .sort_index()
#                 .reindex(range(24), fill_value=0)
#             )

#             # Hitung CV (IDENTIK dengan script Python)
#             cv = (hourly_dist.std() / hourly_dist.mean()) * 100

#             # Interpretasi (IDENTIK)
#             if cv < 50:
#                 is_natural = True
#                 interpretation = (
#                     f"✅ Distribusi waktu MERATA ({cv:.2f}%) → "
#                     f"Pola posting NATURAL, BUKAN bot/buzzer"
#                 )
#             elif cv < 80:
#                 is_natural = True
#                 interpretation = (
#                     f"⚠️ Distribusi waktu CUKUP MERATA ({cv:.2f}%) → "
#                     f"Ada variasi posting, masih wajar"
#                 )
#             else:
#                 is_natural = False
#                 interpretation = (
#                     f"❌ Distribusi waktu TIDAK MERATA ({cv:.2f}%) → "
#                     f"Posting terpusat, kemungkinan bot/buzzer"
#                 )

#             results['temporal_pattern'] = {
#                 'cv': cv,
#                 'is_natural': is_natural,
#                 'hourly_dist': hourly_dist.to_dict(),
#                 'interpretation': interpretation
#             }

#         except Exception:
#             results['temporal_pattern'] = None
#     else:
#         results['temporal_pattern'] = None
    
#     # ========================================
#     # ANALISIS 4: TEXT CHARACTERISTICS
#     # ========================================
#     df_temp = df.copy()
#     df_temp['tweet_length'] = df_temp[text_col].str.len()
#     avg_length = df_temp['tweet_length'].mean()
#     std_length = df_temp['tweet_length'].std()
    
#     df_temp['has_url'] = df_temp[text_col].str.contains(r'http|www', case=False, na=False)
#     df_temp['has_mention'] = df_temp[text_col].str.contains(r'@\w+', na=False)
#     df_temp['has_hashtag'] = df_temp[text_col].str.contains(r'#\w+', na=False)
    
#     url_pct = (df_temp['has_url'].sum() / len(df_temp)) * 100
#     mention_pct = (df_temp['has_mention'].sum() / len(df_temp)) * 100
#     hashtag_pct = (df_temp['has_hashtag'].sum() / len(df_temp)) * 100
    
#     spam_score = 0
#     spam_reasons = []

#     if url_pct > 50:
#         spam_score += 1
#         spam_reasons.append("⚠️ URL terlalu banyak (>50%)")

#     if hashtag_pct > 70:
#         spam_score += 1
#         spam_reasons.append("⚠️ Hashtag terlalu banyak (>70%)")

#     if spam_score == 0:
#         interpretation = (
#             "✅ Karakteristik teks NORMAL\n"
#             "→ Tidak ada indikator spam berdasarkan karakteristik teks"
#         )
#     else:
#         interpretation = (
#             f"❌ Ada {spam_score} indikator spam dari karakteristik teks:\n"
#             + "\n".join(spam_reasons)
#         )

    
#     results['text_characteristics'] = {
#         'avg_length': avg_length,
#         'std_length': std_length,
#         'url_pct': url_pct,
#         'mention_pct': mention_pct,
#         'hashtag_pct': hashtag_pct,
#         'spam_score': spam_score,
#         'spam_reasons': spam_reasons,
#         'interpretation': interpretation
#     }
    
#     # ========================================
#     # ANALISIS 5: HASHTAG DIVERSITY
#     # ========================================
#     all_hashtags = []
#     for text in df[text_col]:
#         hashtags = re.findall(r'#(\w+)', str(text).lower())
#         all_hashtags.extend(hashtags)
    
#     if len(all_hashtags) > 0:
#         total_hashtags = len(all_hashtags)
#         unique_hashtags = len(set(all_hashtags))
#         hashtag_diversity = (unique_hashtags / total_hashtags) * 100
        
#         hashtag_counter = Counter(all_hashtags)
#         top_hashtags = [
#             {
#                 'tag': tag,
#                 'count': count,
#                 'pct': (count / total_hashtags) * 100
#             }
#             for tag, count in hashtag_counter.most_common(15)
#         ]
        
#         top_hashtag_pct = (hashtag_counter.most_common(1)[0][1] / total_hashtags) * 100
#         is_diverse = hashtag_diversity > 50 and top_hashtag_pct < 30
        
#         results['hashtag_diversity'] = {
#             'total_hashtags': total_hashtags,
#             'unique_hashtags': unique_hashtags,
#             'diversity': hashtag_diversity,
#             'top_hashtags': top_hashtags,
#             'top_hashtag_dominance': top_hashtag_pct,
#             'is_diverse': is_diverse,
#             'interpretation': 'Organik' if is_diverse else 'Potensi campaign'
#         }
#     else:
#         results['hashtag_diversity'] = {
#             'total_hashtags': 0,
#             'unique_hashtags': 0,
#             'diversity': 0,
#             'top_hashtags': [],
#             'top_hashtag_dominance': 0,
#             'is_diverse': True,
#             'interpretation': 'Tidak ada hashtag'
#         }
    
#     # ========================================
#     # ANALISIS 6: SENTIMENT VARIATION
#     # ========================================
#     positive_words = [
#         'bagus', 'baik', 'senang', 'suka', 'mantap', 'keren', 'hebat', 'sukses',
#         'terima kasih', 'thanks', 'love', 'good', 'great', 'excellent', 'amazing',
#         'wonderful', 'fantastic', 'perfect', 'best', 'happy', 'setuju', 'support'
#     ]
    
#     negative_words = [
#         'buruk', 'jelek', 'benci', 'tidak suka', 'gagal', 'salah', 'tolak',
#         'bad', 'worst', 'hate', 'terrible', 'awful', 'poor', 'sad', 'angry',
#         'menolak', 'protes', 'korupsi', 'bohong', 'tipu'
#     ]
    
#     neutral_words = [
#         'mungkin', 'sepertinya', 'kira', 'rasa', 'maybe', 'perhaps', 'think',
#         'biasa', 'aja', 'saja', 'cukup', 'lumayan'
#     ]
    
#     def detect_sentiment(text):
#         text_lower = str(text).lower()
        
#         pos_count = sum(1 for word in positive_words if word in text_lower)
#         neg_count = sum(1 for word in negative_words if word in text_lower)
#         neu_count = sum(1 for word in neutral_words if word in text_lower)
        
#         if pos_count > neg_count and pos_count > neu_count:
#             return 'positive'
#         elif neg_count > pos_count and neg_count > neu_count:
#             return 'negative'
#         else:
#             return 'neutral'
    
#     df_temp['sentiment'] = df_temp[text_col].apply(detect_sentiment)
#     sentiment_dist = df_temp['sentiment'].value_counts()
#     sentiment_pct = (sentiment_dist / len(df_temp)) * 100
    
#     # Hitung entropy
#     probabilities = sentiment_dist / len(df_temp)
#     entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
#     max_entropy = np.log2(len(sentiment_dist)) if len(sentiment_dist) > 0 else 1
#     normalized_entropy = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
    
#     dominant_sentiment_pct = sentiment_pct.max()
#     is_diverse = normalized_entropy > 60 and dominant_sentiment_pct < 70
    
#     results['sentiment'] = {
#         'distribution': sentiment_dist.to_dict(),
#         'distribution_pct': sentiment_pct.to_dict(),
#         'diversity_score': normalized_entropy,
#         'dominant_sentiment_pct': dominant_sentiment_pct,
#         'is_diverse': is_diverse,
#         'interpretation': 'Natural' if is_diverse else 'Potensi buzzer'
#     }
    
#     # ========================================
#     # FINAL REPORT
#     # ========================================
#     total_checks = 0
#     passed_checks = 0
    
#     checks = [
#         ('content_diversity', results.get('content_diversity', {})),
#         ('user_behavior', results.get('user_behavior', {})),
#         ('temporal_pattern', results.get('temporal_pattern', {})),
#         ('text_characteristics', results.get('text_characteristics', {})),
#         ('hashtag_diversity', results.get('hashtag_diversity', {})),
#         ('sentiment', results.get('sentiment', {}))
#     ]
    
#     for name, indicator in checks:
#         if indicator:
#             total_checks += 1
#             is_clean = indicator.get('is_diverse') or indicator.get('is_organic') or indicator.get('is_natural')
            
#             # Special case for text_characteristics
#             if name == 'text_characteristics':
#                 is_clean = indicator.get('spam_score', 1) == 0
            
#             if is_clean:
#                 passed_checks += 1
    
#     pass_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
#     results['final_report'] = {
#         'total_checks': total_checks,
#         'passed_checks': passed_checks,
#         'pass_rate': pass_rate
#     }
    
#     return results

def show_preprocessing_steps():
    """Menampilkan langkah-langkah preprocessing"""
    st.header("Preprocessing Steps")
    
    st.markdown("""
    ### Tahapan Preprocessing Data:
    
    #### 1. **Seleksi Kolom Relevan**
    - Mengambil kolom: id, tanggal, tweet/full_text, like, quote, reply, retweet
    - Mengambil kolom user: username, followers, following, verified_status
    
    #### 2. **Rename Kolom**
    - Standarisasi nama kolom untuk konsistensi
    - `id_str` → `id`
    - `full_text` → `tweet` 
    - `favorite_count` → `like`
    - `quote_count` → `quote`
    - `reply_count` → `reply`
    - `retweet_count` → `retweet`
                
    #### 3. **Menambahkan Kolom Baru**
    - Menambahkan kolom: `mentions_count`, `hashtag_count`, `hashtags_list`
                
    #### 4. **Pembersihan Text**
    - Menghapus karakter non-alfabet (simbol, angka berlebih)
    - Menghapus URL
    - Normalisasi Emoji
                
    #### 5. **Filter Bahasa Hindi**
    - Deteksi script Devanagari (Unicode U+0900-U+097F)
    - Deteksi kata-kata Hindi umum
    - Menghapus tweet bahasa Hindi
        
    #### 6. **Filter Bahasa dengan Smart Detection**
    - Deteksi kata-kata khas Indonesia
    - Deteksi kata-kata khas Inggris
    - Menggunakan langdetect sebagai fallback
    - Keep: ID, EN, mixed, uncertain_but_keep
    - Drop: other languages (non ID/EN)
    
    #### 7. **Translate ke Bahasa Indonesia**
    - Translate tweet Bahasa Inggris dan mixed ke Indonesia
    - Menggunakan Google Translator API dengan cache
    - Retry mechanism untuk handling rate limit
    
    #### 8. **Panjang Tweet**
    - Menghitung panjang karakter tweet setelah cleaning
    - Menyimpan dalam kolom `tweet_len`
    
    #### 9. **Hapus Data Kosong**
    - Menghapus baris dengan tweet kosong setelah cleaning
    - Menghapus baris dengan missing values pada kolom penting
    """)
    
    # Tampilkan contoh transformasi
    st.subheader("Contoh Transformasi Text")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Before Cleaning:**")
        st.text_area(
            "",
            "🔥 PROMO HARI INI!!! 🎉\nDiskon 50% untuk semua produk! 💸\nKunjungi website kami di https://example.com\n@ShopeeID #PromoShopee #DiskonBesar",
            height=150,
            disabled=True,
            key="before_clean"
        )
    
    with col2:
        st.markdown("**After Cleaning:**")
        st.text_area(
            "",
            "api promo hari ini diskon untuk semua produk kunjungi website kami",
            height=150,
            disabled=True,
            key="after_clean"
        )
        st.caption("Mentions: 1 | Hashtags: 2 | Language: ID")

def extract_mentions(text):
    """Ekstraksi mentions dari tweet"""
    mentions = re.findall(r'@\w+', text)
    return mentions, len(mentions)

def extract_hashtags(text):
    """Ekstraksi hashtags dari tweet"""
    hashtags = re.findall(r'#\w+', text)
    return hashtags

def clean_text(text):
    """Membersihkan text tweet"""
    # Remove mentions
    # text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    # text = re.sub(r'#\w+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Normalize emoji
    text = emoji.demojize(text)
    text = re.sub(r':(\w+_face|_\w+):', r'[emoji_\1]', text)  # Menambahkan label [emoji_<label>] untuk jenis emoji tertentu

    # Remove extra whitespace
    # text = ' '.join(text.split())
    
    return text

def contains_hindi_script(text):
    """Mendeteksi script Devanagari (Hindi)"""
    if pd.isna(text):
        return False
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(hindi_pattern.search(str(text)))

def detect_hindi_language(text):
    """
    Mendeteksi bahasa menggunakan langdetect library
    """
    if not LANGDETECT_AVAILABLE:
        return False

    if pd.isna(text) or str(text).strip() == '':
        return False

    try:
        detected_lang = detect(str(text))
        # 'hi' adalah kode untuk bahasa Hindi
        return detected_lang == 'hi'
    except (LangDetectException, Exception):
        return False

def detect_hindi_words(text):
    """Deteksi kata-kata Hindi umum"""
    if pd.isna(text):
        return False
    
    hindi_words = {
        'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 'aur',
        'yeh', 'woh', 'kya', 'kaise', 'kahan', 'kab', 'kyun', 'jo', 'agar',
        'tha', 'thi', 'the', 'ho', 'hoga', 'hogi', 'honge', 'kar', 'kiya',
        'karte', 'karti', 'karna', 'maine', 'tumne', 'usne', 'hamne',
        'aap', 'tum', 'main', 'hum', 'voh', 'ye', 'iske', 'uske', 'mere',
        'tere', 'hamare', 'tumhare', 'unke', 'sabko', 'sabke', 'sab'
    }
    
    text_lower = str(text).lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    hindi_word_count = sum(1 for word in words if word in hindi_words)
    
    if len(words) > 0:
        hindi_ratio = hindi_word_count / len(words)
        return hindi_ratio > 0.2
    
    return False

def is_hindi_tweet(text):
    """Kombinasi deteksi script dan kata-kata Hindi"""
    lang_detection = detect_hindi_language(text) if LANGDETECT_AVAILABLE else False

    return contains_hindi_script(text) or lang_detection or detect_hindi_words(text)

# ============================================
# SMART LANGUAGE DETECTION (SAMA DENGAN NOTEBOOK)
# ============================================

def contains_indonesian_words(text):
    """Cek kata-kata khas Indonesia"""
    indonesian_indicators = [
        # Kata umum
        'yang', 'dan', 'ini', 'itu', 'untuk', 'dengan', 'dari', 'tidak', 'ada', 'akan',
        'sudah', 'bisa', 'hanya', 'juga', 'tapi', 'atau', 'pada', 'adalah', 'oleh', 'telah',
        'masih', 'saya', 'anda', 'kita', 'mereka', 'dia', 'kami',
        # Kata slang/informal
        'gak', 'udah', 'udh', 'aja', 'banget', 'bgt', 'dong', 'deh', 'sih', 'nih',
        'kalo', 'gue', 'gw', 'lo', 'lu', 'wkwk', 'wkwkwk', 'kwkw', 'anjir', 'anjay',
        'mantap', 'mantul', 'keren', 'asik', 'oke', 'siap', 'gas', 'cuy', 'bro', 'gan',
        # Kata promosi/commerce
        'diskon', 'gratis', 'promo', 'voucher', 'cashback', 'murah', 'sale', 'cod',
        'shopee', 'tokopedia', 'lazada', 'bukalapak', 'blibli', 'gopay', 'ovo', 'dana',
        'bayar', 'beli', 'jual', 'harga', 'kode', 'claim', 'dapat', 'hingga', 'sampai',
        # Kata tanya
        'apa', 'siapa', 'dimana', 'kemana', 'kapan', 'kenapa', 'bagaimana', 'gimana',
        # Partikel
        'lah', 'kah', 'nya'
    ]
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    indo_word_count = sum(1 for word in words if word in indonesian_indicators)
    
    return indo_word_count

def contains_english_words(text):
    """Cek kata-kata khas Inggris"""
    english_indicators = [
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was',
        'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may',
        'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'let', 'put',
        'say', 'she', 'too', 'use', 'will', 'with', 'have', 'this', 'that', 'from',
        'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'there',
        'would', 'make', 'like', 'into', 'time', 'look', 'only', 'come', 'over',
        'think', 'also', 'back', 'after', 'just', 'where', 'most', 'know', 'than',
        'want', 'keep', 'eye', 'something', 'until', 'gonna', 'wanna', 'gotta'
    ]
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    eng_word_count = sum(1 for word in words if word in english_indicators)
    
    return eng_word_count

def is_meaningful_text(text):
    """Cek apakah teks punya cukup huruf"""
    letter_count = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(re.sub(r'\s', '',text))
    
    if total_chars == 0:
        return False
    
    letter_ratio = letter_count / total_chars
    return letter_ratio >= 0.3

def preprocess_for_detection(text):
    """Preprocessing ringan untuk deteksi bahasa"""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def detect_language_smart(text, threshold=0.6):
    """Deteksi bahasa dengan multiple fallback mechanism"""
    # if not TRANSLATION_AVAILABLE:
    #     return 'id'  # Default ke Indonesia jika library tidak tersedia
    
    original_text = str(text)
    
    #Cek apakah teks meaningful
    if not is_meaningful_text(text):
        return 'noisy'
    
    # Preprocessing
    clean_text = preprocess_for_detection(text)
    
    if len(clean_text.strip()) < 3:
        return 'too_short'
    
    # Rule-based: cek kata-kata khas
    indo_words = contains_indonesian_words(original_text)
    eng_words = contains_english_words(original_text)
    
    # Jika ada kata Indonesia yang cukup banyak, langsung keep
    if indo_words >= 2:
        return 'id'
    
    # Jika ada kata Inggris yang cukup banyak, langsung keep
    if eng_words >= 2:
        return 'en'
    
    # Jika ada minimal 1 kata Indo atau Inggris, keep as mixed
    if indo_words >= 1 or eng_words >= 1:
        return 'mixed'
    
    # Fallback ke langdetect
    try:
        langs = detect_langs(clean_text)
        
        id_conf = next((lang.prob for lang in langs if lang.lang == 'id'), 0)
        en_conf = next((lang.prob for lang in langs if lang.lang == 'en'), 0)
        
        if id_conf >= threshold:
            return 'id'
        elif en_conf >= threshold:
            return 'en'
        elif (id_conf + en_conf) >= 0.5:
            return 'mixed'
        else:
            top_lang = langs[0].lang
            if top_lang not in ['id', 'en']:
                return 'other'
            else:
                return top_lang
    
    except LangDetectException:
        # Jika langdetect gagal, tapi punya kata Indo/Eng sedikit, tetap keep
        if indo_words >= 1 or eng_words >= 1:
            return 'uncertain_but_keep'
        return 'uncertain'

# ============================================
# TRANSLATION WITH CACHE (SAMA DENGAN NOTEBOOK)
# ============================================

# Global cache untuk translation
translation_cache = {}

def clean_text_for_translation(text):
    """Cleaning ringan untuk translation"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def translate_fast(text, max_retries=2):
    """Translate dengan retry mechanism"""
    if not TRANSLATION_AVAILABLE:
        return text
    
    text = clean_text_for_translation(text)
    
    if text == "":
        return text
    
    translator = GoogleTranslator(source='en', target='id')
    
    for _ in range(max_retries):
        try:
            return translator.translate(text)
        except:
            time.sleep(1)
    
    # return translator.translate(text)
    return text

def translate_with_cache(text):
    """Translate dengan cache untuk efisiensi"""
    if text in translation_cache:
        return translation_cache[text]
    
    translated = translate_fast(text)
    translation_cache[text] = translated
    return translated

# ============================================
# MAIN PREPROCESSING FUNCTION
# ============================================

def preprocess_dataframe(df, progress_callback=None):
    total_steps = 14  # Updated total steps
    current_step = 0
    
    def update_progress(message):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, message)
    
    # Step 1: Format tanggal
    update_progress("Memformat kolom tanggal...")
    if 'created_at' in df.columns:
        df['tanggal'] = pd.to_datetime(df['created_at'], errors='coerce')
    elif 'tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    
    # Step 2: Hapus duplikat berdasarkan ID
    # update_progress("Menghapus data duplikat...")
    # id_col = 'id_str' if 'id_str' in df.columns else 'id' if 'id' in df.columns else None
    # if id_col:
    #     df = df.drop_duplicates(subset=[id_col], keep='first')
    
    # Step 3: Rename kolom
    update_progress("Mengganti nama kolom...")
    rename_dict = {
        'favorite_count': 'like',
        'quote_count': 'quote',
        'reply_count': 'reply',
        'retweet_count': 'retweet',
        'full_text': 'tweet',
        'id_str': 'id'
    }
    df = df.rename(columns=rename_dict)
    
    # Step 4: Pilih kolom relevan
    update_progress("Memilih kolom relevan...")
    required_cols = ['id', 'tanggal', 'tweet', 'like', 'quote', 'reply', 'retweet', 
                     'username', 'followers', 'following', 'verified_status']
    
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Step 5: Ekstraksi mentions
    update_progress("Mengekstraksi mentions...")
    #df['mentions_list'] = df['tweet'].apply(lambda x: extract_mentions(x)[0])
    df['mentions_count'] = df['tweet'].apply(lambda x: extract_mentions(x)[1])
    
    # Step 6: Ekstraksi hashtags
    update_progress("Mengekstraksi hashtags...")
    df['hashtags_list'] = df['tweet'].apply(extract_hashtags)
    df['hashtags_count'] = df['hashtags_list'].apply(len)
    
    # Step 7: Clean text
    update_progress("Membersihkan text...")
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)
    
    # Step 8: Filter Hindi tweets
    update_progress("Menghapus tweet bahasa Hindi...")
    if TRANSLATION_AVAILABLE:
        df['is_hindi'] = df['cleaned_tweet'].apply(is_hindi_tweet)
        hindi_count = df['is_hindi'].sum()
        df = df[~df['is_hindi']].copy()
        df = df.drop(columns=['is_hindi'])
    else:
        hindi_count = 0
    
    # Step 9: Smart language detection
    update_progress("Mendeteksi bahasa dengan smart detection...")
    if TRANSLATION_AVAILABLE:
        df['detected_lang'] = df['cleaned_tweet'].apply(
            lambda x: detect_language_smart(str(x), threshold=0.6)
        )
        
        # Step 10: Filter bahasa
        update_progress("Memfilter tweet bahasa selain ID/EN...")
        
        # Keep: id, en, mixed, uncertain_but_keep, too_short, noisy
        keep_langs = ['id', 'en', 'mixed', 'uncertain_but_keep', 'too_short', 'noisy', 'uncertain']
        drop_langs = ['other']
        
        before_filter = len(df)
        df = df[df['detected_lang'].isin(keep_langs)].copy()
        after_filter = len(df)
        dropped_count = before_filter - after_filter
        
        # Step 11: Translate 
        update_progress("Menerjemahkan tweet EN + MIXED ke Indonesia...")
        
        # Set default
        df['final_tweet'] = df['cleaned_tweet']
        
        # Translate yang perlu
        mask_translate = df['detected_lang'].isin(['en', 'mixed'])
        translate_count = mask_translate.sum()
        
        if translate_count > 0:
            print(f"Translating {translate_count} tweets...")
            df.loc[mask_translate, 'final_tweet'] = (
                df.loc[mask_translate, 'cleaned_tweet'].progress_apply(translate_with_cache)
            )
        
        # Drop kolom detected_lang (opsional)
        df = df.drop(columns=['detected_lang'])
        
    else:
        df['final_tweet'] = df['cleaned_tweet']
        update_progress("Melewati deteksi bahasa (library tidak tersedia)...")
        update_progress("Melewati filter bahasa...")
        update_progress("Melewati translate...")
        hindi_count = 0
        dropped_count = 0
        translate_count = 0
    
    # Step 12: Hitung panjang tweet
    update_progress("Menghitung panjang tweet...")
    df['tweet_len'] = df['final_tweet'].apply(lambda x: len(str(x)))
    
    # Step 13: Hapus data kosong
    update_progress("Menghapus data kosong...")
    df = df[df['final_tweet'].str.strip() != ''].copy()
    df = df.dropna(subset=['final_tweet'])
    
    # Step 14: Reorder kolom
    update_progress("Finalisasi kolom output...")
    final_cols = ['id', 'tanggal', 'final_tweet', 'tweet_len', 'like', 'quote', 
                  'reply', 'retweet', 'username', 'followers', 'following', 
                  'verified_status', 'mentions_count', 'hashtags_count', 'hashtags_list']
    
    # Tambahkan detected_lang jika ada
    if 'detected_lang' in df.columns:
        final_cols.insert(3, 'detected_lang')
    
    available_final_cols = [col for col in final_cols if col in df.columns]
    df = df[available_final_cols]
    
    # Store statistics
    df.attrs['preprocessing_stats'] = {
        'hindi_removed': hindi_count,
        'other_lang_removed': dropped_count,
        'translated': translate_count
    }
    
    return df

def show_upload_data():
    """Upload dan preprocessing data baru"""
    st.header("Upload & Process Data")
    
    st.markdown("""
    ### Petunjuk Upload Data:
    
    **Format File**: Excel (.xlsx) atau CSV (.csv)
    
    **Kolom yang Diperlukan**:
    - `id` atau `id_str`: ID unik tweet
    - `tanggal` atau `created_at`: Timestamp tweet
    - `tweet` atau `full_text`: Konten tweet
    - `like` atau `favorite_count`: Jumlah like
    - `quote` atau `quote_count`: Jumlah quote
    - `reply` atau `reply_count`: Jumlah reply
    - `retweet` atau `retweet_count`: Jumlah retweet
    - `username`: Username akun
    - `followers`: Jumlah followers
    - `following`: Jumlah following
    - `verified_status`: Status verifikasi
    """)
    
    uploaded_file = st.file_uploader("Upload Raw Data atau Merged Data", type=['xlsx', 'csv'], key='process_upload')
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File berhasil diupload! Total rows: {len(df_raw)}")
            
            # Preview
            st.subheader("Preview Raw Data")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            # Show columns
            st.subheader("Kolom yang Terdeteksi")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Kolom dalam File:**")
                st.write(list(df_raw.columns))
            
            with col2:
                st.write("**Statistik Data:**")
                st.write(f"- Total Rows: {len(df_raw)}")
                st.write(f"- Total Columns: {len(df_raw.columns)}")
                st.write("**Missing Values:**")
                for col, val in df_raw.isnull().sum().items():
                    if val > 0:
                        st.write(f"- `{col}`: {val}")
               
            # Check if translation is available
            if not TRANSLATION_AVAILABLE:
                st.warning("""
                ⚠️ **Library untuk deteksi bahasa dan translate tidak tersedia**
                
                Proses akan tetap berjalan, namun:
                - Deteksi bahasa akan dilewati
                - Filter bahasa akan dilewati
                - Translate Bahasa Inggris ke Indonesia akan dilewati
                
                Untuk mengaktifkan fitur ini, install library berikut:
                ```
                pip install langdetect deep-translator
                ```
                """)
            
            # Button untuk preprocessing
            if st.button("Jalankan Preprocessing", type="primary", use_container_width=True):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    # Jalankan preprocessing
                    with st.spinner("Memproses data..."):
                        df_processed = preprocess_dataframe(df_raw, update_progress)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Get statistics
                    stats = df_processed.attrs.get('preprocessing_stats', {})
                    hindi_removed = stats.get('hindi_removed', 0)
                    other_removed = stats.get('other_lang_removed', 0)
                    translated = stats.get('translated', 0)
                    
                    st.success(f"""
                    ✅ Preprocessing selesai!
                    - Data awal: {len(df_raw)} rows
                    - Tweet Hindi dihapus: {hindi_removed}
                    - Tweet bahasa lain dihapus: {other_removed}
                    - Tweet diterjemahkan: {translated}
                    - **Data final: {len(df_processed)} rows**
                    """)
                    
                    # Show hasil
                    st.subheader("Preview Data Hasil Preprocessing")
                    st.dataframe(df_processed.head(10), use_container_width=True)
                    
                    # Statistik
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", len(df_processed))
                    with col2:
                        avg_len = df_processed['tweet_len'].mean() if 'tweet_len' in df_processed.columns else 0
                        st.metric("Avg Tweet Length", f"{avg_len:.0f}")
                    with col3:
                        avg_hashtags = df_processed['hashtags_count'].mean() if 'hashtags_count' in df_processed.columns else 0
                        st.metric("Avg Hashtags", f"{avg_hashtags:.1f}")
                    with col4:
                        avg_mentions = df_processed['mentions_count'].mean() if 'mentions_count' in df_processed.columns else 0
                        st.metric("Avg Mentions", f"{avg_mentions:.1f}")
                    
                    # Language distribution (jika ada)
                    if 'detected_lang' in df_processed.columns:
                        st.subheader("Distribusi Bahasa Terdeteksi")
                        lang_dist = df_processed['detected_lang'].value_counts()
                        st.dataframe(lang_dist.to_frame('count'), use_container_width=True)
                    
                    # Store di session state
                    st.session_state['df_processed'] = df_processed
                    
                    # Download section
                    st.markdown("---")
                    st.subheader("Download Hasil Preprocessing")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download as Excel
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df_processed.to_excel(writer, index=False, sheet_name='Data Bersih')
                        buffer.seek(0)
                        
                        st.download_button(
                            label="Download Excel (.xlsx)",
                            data=buffer,
                            file_name="data_bersih.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download as CSV
                        csv_buffer = df_processed.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="Download CSV (.csv)",
                            data=csv_buffer,
                            file_name="data_bersih.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    st.info("File hasil preprocessing siap digunakan untuk modeling atau analisis lebih lanjut!")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error saat preprocessing: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")
    else:
        st.info("Upload file untuk memulai proses preprocessing")

def show_visualizations():
    """Menampilkan visualisasi data"""
    st.header("Visualisasi Data")
    
    # Cek apakah ada data di session state
    df = None

    if df is None:
        st.subheader("Upload Data (CSV / Excel)")
        uploaded_file = st.file_uploader(
            "Upload file hasil preprocessing",
            type=["csv", "xlsx"],
            help="Upload file CSV atau Excel untuk melihat visualisasi"
        )

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ File `{uploaded_file.name}` berhasil dimuat")
            st.caption(f"Total baris: {len(df)} | Total kolom: {len(df.columns)}")

        except Exception as e:
            # st.error(f"❌ Gagal membaca file: {str(e)}")
            return
    
    # Distribution of engagement metrics
    st.subheader("Distribusi Engagement Metrics")
    
    col1, col2 = st.columns(2)
    
    if 'like' in df.columns:
        with col1:
            fig = px.histogram(df, x='like', nbins=50, 
                              title='Distribusi Jumlah Like',
                              labels={'like': 'Jumlah Like'})
            st.plotly_chart(fig, use_container_width=True)
    
    if 'retweet' in df.columns:
        with col2:
            fig = px.histogram(df, x='retweet', nbins=50,
                              title='Distribusi Jumlah Retweet',
                              labels={'retweet': 'Jumlah Retweet'})
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    if 'reply' in df.columns:
        with col3:
            fig = px.histogram(
                df, x='reply', nbins=50,
                title='Distribusi Jumlah Reply',
                labels={'reply': 'Jumlah Reply'}
            )
            st.plotly_chart(fig, use_container_width=True)

    if 'quote' in df.columns:
        with col4:
            fig = px.histogram(
                df, x='quote', nbins=50,
                title='Distribusi Jumlah Quote',
                labels={'quote': 'Jumlah Quote'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
    # Tweet length distribution
    if 'tweet_len' in df.columns:
        st.subheader("Distribusi Panjang Tweet")
        fig = px.histogram(df, x='tweet_len', nbins=50,
                          title='Distribusi Panjang Tweet',
                          labels={'tweet_len': 'Panjang Tweet (karakter)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Top users by followers
    # if 'username' in df.columns and 'followers' in df.columns:
    #     st.subheader("Top 10 Users by Followers")
    #     top_users = df.groupby('username')['followers'].first().sort_values(ascending=False).head(10)
        
    #     fig = px.bar(x=top_users.values, y=top_users.index, orientation='h',
    #                 title='Top 10 Users by Followers',
    #                 labels={'x': 'Followers', 'y': 'Username'})
    #     st.plotly_chart(fig, use_container_width=True)

    # Time series frekuensi tweet harian
    if 'tanggal' in df.columns:
        st.subheader("Frekuensi Tweet Harian")
        
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        ts = df.groupby(pd.Grouper(key='tanggal', freq='D')).size().reset_index(name='jumlah')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts['tanggal'], y=ts['jumlah'],
            mode='lines', line=dict(color='royalblue', width=1.5)
        ))
        fig.update_layout(
            title='Frekuensi Tweet Harian',
            xaxis_title='Tanggal',
            yaxis_title='Jumlah Tweet',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # WordCloud
    text_col = 'final_tweet' if 'final_tweet' in df.columns else 'tweet' if 'tweet' in df.columns else None
    
    if text_col:
        st.subheader("Word Cloud - Most Common Words")
        
        text = ' '.join(df[text_col].astype(str))
        
        if text.strip():
            try:
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    colormap='viridis').generate(text)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error membuat WordCloud: {str(e)}")
    
    # Hashtag distribution
    if 'hashtags_list' in df.columns:
        st.subheader("Top 10 Hashtags")

        all_hashtags = []

        for val in df['hashtags_list'].dropna():
            # Jika sudah list
            if isinstance(val, list):
                all_hashtags.extend(val)

            # Jika string list: "['tag1','tag2']"
            elif isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        all_hashtags.extend(parsed)
                except:
                    pass

        if not all_hashtags:
            st.info("ℹ️ Tidak ada hashtag valid yang bisa divisualisasikan")
        else:
            hashtag_counts = (
                pd.Series(all_hashtags)
                .str.lower()
                .value_counts()
                .head(10)
            )

            fig = px.bar(
                x=hashtag_counts.values,
                y=hashtag_counts.index,
                orientation='h',
                title='Top 10 Hashtags',
                labels={'x': 'Count', 'y': 'Hashtag'}
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    show()