import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import warnings
import os
from collections import Counter
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)
from bertopic import BERTopic
import pickle

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
FORECAST_DAYS  = 30
TOP_N_ITEMS    = 10
N_TRIALS       = 50
TEST_SIZE      = 0.2
LAG_FEATURES   = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 14]

STOPWORDS_INDONESIA = {
    # Kata sambung
    'dan', 'atau', 'tetapi', 'namun', 'sedangkan', 'padahal', 'serta',
    'maupun', 'melainkan', 'bahwa', 'karena', 'sebab', 'jika', 'kalau', 'ini',

    # Kata depan
    'di', 'ke', 'dari', 'pada', 'untuk', 'dengan', 'dalam', 'oleh',
    'terhadap', 'atas', 'antara', 'kepada', 'bagi', 'tentang',

    # Kata ganti
    'saya', 'aku', 'kamu', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita',
    'nya', 'ku', 'mu', 'kak', 'kakak',

    # Kata kerja bantu
    'adalah', 'ialah', 'merupakan', 'yaitu', 'yakni', 'akan', 'telah',
    'sudah', 'sedang', 'dapat', 'bisa', 'boleh', 'harus', 'perlu',

    # Kata keterangan
    'sangat', 'lebih', 'paling', 'terlalu', 'cukup', 'agak', 'sedikit',
    'banyak', 'semua', 'setiap', 'seluruh', 'beberapa', 'berbagai',

    # Kata tanya
    'apa', 'siapa', 'kapan', 'dimana', 'kemana', 'darimana', 'mengapa',
    'kenapa', 'bagaimana', 'berapa',

    # Kata penunjuk
    'ini', 'itu', 'tersebut', 'begitu', 'begini',

    # Partikel
    'lah', 'kah', 'pun', 'per', 'tah', 'tak', 'nak', 'hai', 'dah',

    # Kata umum lainnya
    'yang', 'juga', 'ada', 'tidak', 'ya', 'bukan', 'belum', 'masih',
    'hanya', 'saja', 'lagi', 'pula', 'jadi', 'maka', 'bila', 'walau',
    'meski', 'walaupun', 'meskipun', 'hingga', 'sampai', 'supaya', 'mau',

    # Twitter specific (optional, sesuaikan dengan kebutuhan)
    'rt', 'via', 'follow', 'followers', 'following', 'retweet', 'reply'
}

BERTOPIC_MODEL_PATH = "models/bertopic(1).pkl"

# ────────────────
# HELPER FUNCTIONS
# ────────────────

def tokenize_text(text):
    if pd.isna(text):
        return []
    # text = str(text).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', str(text))
    return [w for w in words if len(w) >= 3 and w.lower() not in STOPWORDS_INDONESIA]

def extract_hashtags(hashtag_str):
    if pd.isna(hashtag_str) or str(hashtag_str).strip() == '[]':
        return []
    try:
        hashtags = eval(hashtag_str)
        return [h.lower() for h in hashtags if h]
    except:
        return []

def prepare_time_series(df_in, date_col, freq='D'):
    ts = df_in.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name='y')
    ts.columns = ['ds', 'y']
    date_range = pd.date_range(start=ts['ds'].min(), end=ts['ds'].max(), freq=freq)
    full = pd.DataFrame({'ds': date_range})
    ts = full.merge(ts, on='ds', how='left').fillna(0)
    return ts

def create_time_features(df_in):
    df_in = df_in.copy()
    df_in['dayofweek']  = df_in['ds'].dt.dayofweek
    df_in['dayofmonth'] = df_in['ds'].dt.day
    df_in['weekofyear'] = df_in['ds'].dt.isocalendar().week.astype(int)
    df_in['month']      = df_in['ds'].dt.month
    df_in['is_weekend'] = (df_in['dayofweek'] >= 5).astype(int)
    return df_in

def create_lag_features(df_in, target_col='y'):
    df_in = df_in.copy()
    for lag in LAG_FEATURES:
        df_in[f'lag_{lag}'] = df_in[target_col].shift(lag)
    return df_in

def create_rolling_features(df_in, target_col='y'):
    df_in = df_in.copy()
    for w in ROLLING_WINDOWS:
        base = df_in[target_col].shift(1)
        df_in[f'rolling_mean_{w}'] = base.rolling(w).mean()
        df_in[f'rolling_std_{w}']  = base.rolling(w).std()
        df_in[f'rolling_min_{w}']  = base.rolling(w).min()
        df_in[f'rolling_max_{w}']  = base.rolling(w).max()
    return df_in

def prepare_features(ts_df):
    df_f = ts_df.copy()
    df_f = create_time_features(df_f)
    df_f = create_lag_features(df_f)
    df_f = create_rolling_features(df_f)
    df_f = df_f.dropna()
    return df_f

def rmsse(y_true, y_pred, y_train):
    denom = np.mean(np.diff(y_train) ** 2)
    if denom == 0:
        return np.nan
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / denom)

def calculate_metrics(y_true, y_pred, y_train):
    y_pred = np.maximum(y_pred, 0)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsse_val = rmsse(y_true, y_pred, y_train)
    return {'MAE': mae, 'RMSE': rmse, 'RMSSE': rmsse_val}

def forecast_with_lgbm(ts_df, item_name, progress_text=None):
    df_features = prepare_features(ts_df)
    if len(df_features) < 20:
        return None

    feature_cols = [c for c in df_features.columns if c not in ['ds', 'y']]
    test_size    = int(len(df_features) * TEST_SIZE)
    train_df     = df_features.iloc[:-test_size].copy()
    test_df      = df_features.iloc[-test_size:].copy()

    val_size = int(len(train_df) * 0.2)
    X_tr = train_df.iloc[:-val_size][feature_cols].values
    y_tr = train_df.iloc[:-val_size]['y'].values
    X_vl = train_df.iloc[-val_size:][feature_cols].values
    y_vl = train_df.iloc[-val_size:]['y'].values

    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse',
            'verbosity': -1, 'boosting_type': 'gbdt',
            'num_leaves':       trial.suggest_int('num_leaves', 20, 150),
            'max_depth':        trial.suggest_int('max_depth', 3, 12),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
            'min_child_samples':trial.suggest_int('min_child_samples', 5, 50),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        try:
            from sklearn.metrics import mean_squared_error as mse_fn
            m = lgb.LGBMRegressor(**params, random_state=42)
            m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
            return np.sqrt(mse_fn(y_vl, m.predict(X_vl)))
        except:
            return float('inf')

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = {
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'boosting_type': 'gbdt', 'random_state': 42,
        **study.best_params
    }

    # Evaluasi pada test set
    X_tr_full = train_df[feature_cols].values
    y_tr_full = train_df['y'].values
    X_te      = test_df[feature_cols].values
    y_te      = test_df['y'].values

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_tr_full, y_tr_full)
    y_pred_test  = final_model.predict(X_te)
    test_metrics = calculate_metrics(y_te, y_pred_test, y_tr_full)

    # Retrain pada semua data
    X_full = df_features[feature_cols].values
    y_full = df_features['y'].values
    final_model.fit(X_full, y_full)

    # Forecast ke depan
    current_df       = ts_df.copy()
    future_preds     = []
    for _ in range(FORECAST_DAYS):
        next_date = current_df['ds'].max() + pd.Timedelta(days=1)
        current_df = pd.concat(
            [current_df, pd.DataFrame({'ds': [next_date], 'y': [0]})],
            ignore_index=True
        )
        feats = prepare_features(current_df)
        if len(feats) == 0:
            break
        pred = max(0, final_model.predict(feats.iloc[-1:][feature_cols].values)[0])
        future_preds.append(pred)
        current_df.loc[current_df.index[-1], 'y'] = pred

    last_date    = ts_df['ds'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(future_preds), freq='D'
    )

    return {
        'item_name':      item_name,
        'best_params':    study.best_params,
        'best_rmse':      study.best_value,
        'test_metrics':   test_metrics,
        'train_df':       train_df,
        'test_df':        test_df,
        'test_predictions': y_pred_test,
        'forecast_df':    pd.DataFrame({'ds': future_dates, 'y_pred': future_preds}),
        'full_ts':        ts_df,
        'feature_importance': pd.DataFrame({
            'feature':    feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
    }

# ─────────────────────────────────────────────
# LOAD BERTOPIC (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model BERTopic...")
def load_bertopic_model():
    if not os.path.exists(BERTOPIC_MODEL_PATH):
        return None
    return BERTopic.load(BERTOPIC_MODEL_PATH)

# ─────────────────────────────────────────────
# TOPIC EXTRACTION
# ─────────────────────────────────────────────
def get_topic_name(topic_id, model):
    if topic_id == -1:
        return "Outlier"
    words = model.get_topic(topic_id)
    if words:
        return f"Topic_{topic_id}: {', '.join([w for w, _ in words[:3]])}"
    return f"Topic_{topic_id}"

def extract_topics(df, topic_model):
    keywords = r'(?i)\b(promo|diskon|sale|flash\s*sale|amp)\b'
    df = df.copy()
    df['final_tweet'] = (
        df['final_tweet'].astype(str)
        .str.replace(keywords, '', regex=True)
        .str.replace(r'\b\w{1,2}\b', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    tweets = df['final_tweet'].tolist()

    with st.spinner("Encoding tweets (IndoBERTweet)..."):
        emb_model  = topic_model.embedding_model.embedding_model
        embeddings = emb_model.encode(tweets, show_progress_bar=False)

    with st.spinner("Mengekstraksi topik..."):
        topics, probs = topic_model.transform(tweets, embeddings)

    topic_probs = []
    for i, tid in enumerate(topics):
        topic_probs.append(0.0 if tid == -1 else float(probs[i][tid]))

    df['topic_id']          = topics
    df['topic_name']        = [get_topic_name(t, topic_model) for t in topics]
    df['topic_probability'] = topic_probs
    return df

# ─────────────────────────────────────────────
# CHART HELPER
# ─────────────────────────────────────────────
def make_forecast_chart(result, title):
    full_ts   = result['full_ts']
    test_df   = result['test_df']
    fore_df   = result['forecast_df']
    test_pred = result['test_predictions']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full_ts['ds'], y=full_ts['y'],
        mode='lines', name='Historical',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test_df['ds'], y=test_pred,
        mode='lines', name='Test Prediction',
        line=dict(color='orange', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=fore_df['ds'], y=fore_df['y_pred'],
        mode='lines+markers', name=f'Forecast ({FORECAST_DAYS}d)',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    fig.update_layout(
        title=title, xaxis_title='Tanggal', yaxis_title='Frekuensi',
        hovermode='x unified', height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=60)
    )
    return fig

def make_metrics_cols(result):
    m   = result['test_metrics']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE",   f"{m['MAE']:.2f}")
    col2.metric("RMSE",  f"{m['RMSE']:.2f}")
    col3.metric("RMSSE", f"{m['RMSSE']:.3f}")
    col4.metric("Best Optuna RMSE", f"{result['best_rmse']:.3f}")

# ─────────────────────────────────────────────
# MAIN SHOW
# ─────────────────────────────────────────────
def show():
    st.title("Forecasting Frekuensi Topik, Hashtag & Kata")
    st.markdown("---")
    st.header("Proses yang Dilakukan")
    st.markdown("""
    1. Upload data bersih
    2. Ekstraksi topik (BERTopic)
    3. Hitung frekuensi kemunculan Topik, Hashtag, dan Kata
    4. Forecasting (LightGBM+Optuna)
    """)
    st.markdown("---")

    # ── STEP 1: Upload Data ─────────────────────────────────────
    st.header("Upload Data")
    st.markdown("""
    **Kolom yang diperlukan:**
    - `final_tweet`: Teks tweet yang sudah dibersihkan
    - `tanggal`: Timestamp tweet
    - `hashtags_list`: List hashtag
    """)
    uploaded = st.file_uploader(
        "Upload file Excel (.xlsx) — data yang sudah bersih & melalui preprocessing",
        type=["xlsx"]
    )
    if uploaded is None:
        st.info("Silakan upload file untuk memulai.")
        return

    df_raw = pd.read_excel(uploaded)
    st.success(f"✅ Data berhasil diupload: **{len(df_raw):,} baris**")

    required_cols = ['final_tweet', 'tanggal', 'hashtags_list']
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan: {missing}")
        return

    df_raw['tanggal'] = pd.to_datetime(df_raw['tanggal'])

    with st.expander("Preview data (10 baris pertama)"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("---")

    # ── STEP 2: Ekstraksi Topik BERTopic ────────────────────────
    st.header("Ekstraksi Topik dengan BERTopic")

    topic_model = load_bertopic_model()
    if topic_model is None:
        st.error(f"⚠️ Model BERTopic tidak ditemukan di `{BERTOPIC_MODEL_PATH}`. "
                "Pastikan file model sudah tersimpan.")
        return

    st.info(f"Model BERTopic loaded — **{len(topic_model.get_topic_info()) - 1} topik** tersedia.")

    TOPIC_SAVE_PATH = "models/topic_results.pkl"

    col_extract, col_load_topic = st.columns([2, 1])

    with col_load_topic:
        if os.path.exists(TOPIC_SAVE_PATH):
            if st.button("📂 Load Hasil Topik Tersimpan", use_container_width=True):
                with open(TOPIC_SAVE_PATH, "rb") as f:
                    saved_topic = pickle.load(f)
                st.session_state['df_topics'] = saved_topic['df_topics']
                st.success(f"✅ Hasil topik dimuat! (Disimpan: {saved_topic.get('saved_at', '-')})")
        else:
            st.info("Belum ada hasil topik tersimpan.")

    with col_extract:
        if st.button("🚀 Jalankan Ekstraksi Topik", type="primary", use_container_width=True):
            df_topics = extract_topics(df_raw, topic_model)
            st.session_state['df_topics'] = df_topics

            # Auto-save
            os.makedirs("models", exist_ok=True)
            with open(TOPIC_SAVE_PATH, "wb") as f:
                pickle.dump({
                    'df_topics': df_topics,
                    'saved_at':  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)

            st.success("✅ Ekstraksi topik selesai dan otomatis tersimpan!")

    if 'df_topics' not in st.session_state:
        st.warning("Klik tombol di atas untuk mengekstraksi topik terlebih dahulu.")
        return

    df = st.session_state['df_topics'].copy()

    # ── Topik yang Terbentuk ─────────────────────────────────────
    st.subheader("Topik yang Terbentuk")

    df_no_outlier_topics = df[df['topic_id'] != -1].copy()

    topic_counts = (
        df_no_outlier_topics
        .groupby(['topic_id', 'topic_name'])
        .size()
        .reset_index(name='jumlah_tweet')
        .sort_values('jumlah_tweet', ascending=False)
        .reset_index(drop=True)
    )
    topic_counts.index += 1
    topic_counts.index.name = 'Rank'

    st.dataframe(
        topic_counts.rename(columns={
            'topic_id':    'Topic ID',
            'topic_name':  'Nama Topik',
            'jumlah_tweet':'Jumlah Tweet'
        }),
        use_container_width=True
    )

    st.markdown("---")

    # ── Contoh Tweet per Topik ───────────────────────────────────
    st.subheader("Contoh Tweet per Topik")

    N_SAMPLES = 3
    unique_topics = sorted(df_no_outlier_topics['topic_id'].unique())

    for topic_id in unique_topics:
        topic_data = (
            df_no_outlier_topics[df_no_outlier_topics['topic_id'] == topic_id]
            .sort_values('topic_probability', ascending=False)
        )
        topic_name = topic_data['topic_name'].iloc[0]
        tweet_count = len(topic_data)

        with st.expander(f"**{topic_name}** — {tweet_count:,} tweet", expanded=False):
            samples = topic_data.head(N_SAMPLES)
            for i, row in enumerate(samples.itertuples(), 1):
                st.markdown(
                    f"""
                    <div style="background:#f8f9fa;border-left:4px solid #4A90D9;
                                padding:10px 14px;border-radius:4px;margin-bottom:8px;">
                        <span style="color:#888;font-size:0.8rem;">#{i} · prob: {row.topic_probability:.3f}</span><br>
                        <span style="font-size:0.95rem;">{row.final_tweet}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── STEP 3: Ekstraksi Frekuensi ─────────────────────────────
    st.header("Analisis Frekuensi Top 10")

    # --- Topik ---
    df_no_outlier = df[df['topic_name'] != 'Outlier'].copy()
    topic_freq    = df_no_outlier['topic_name'].value_counts()
    top_topics    = topic_freq.head(TOP_N_ITEMS).index.tolist()

    # --- Kata ---
    all_words_dates = []
    for _, row in df.iterrows():
        for w in tokenize_text(row['final_tweet']):
            all_words_dates.append({'tanggal': row['tanggal'], 'kata': w.lower()})
    words_df   = pd.DataFrame(all_words_dates)
    word_freq  = words_df['kata'].value_counts()
    top_words  = word_freq.head(TOP_N_ITEMS).index.tolist()

    # --- Hashtag ---
    all_ht_dates = []
    for _, row in df.iterrows():
        for h in extract_hashtags(row['hashtags_list']):
            all_ht_dates.append({'tanggal': row['tanggal'], 'hashtag': h})
    hashtags_df  = pd.DataFrame(all_ht_dates)
    hashtag_freq = hashtags_df['hashtag'].value_counts() if len(hashtags_df) > 0 else pd.Series(dtype=int)
    top_hashtags = hashtag_freq.head(TOP_N_ITEMS).index.tolist()

    tab_t, tab_w, tab_h = st.tabs(["Top 10 Topik", "Top 10 Kata", "Top 10 Hashtag"])

    with tab_t:
        df_tv = pd.DataFrame({'Rank': range(1, len(top_topics)+1),
                              'Topik': top_topics,
                              'Frekuensi': [topic_freq[t] for t in top_topics]})
        st.dataframe(df_tv, use_container_width=True, hide_index=True)

    with tab_w:
        df_wv = pd.DataFrame({'Rank': range(1, len(top_words)+1),
                              'Kata': top_words,
                              'Frekuensi': [word_freq[w] for w in top_words]})
        st.dataframe(df_wv, use_container_width=True, hide_index=True)

    with tab_h:
        if top_hashtags:
            df_hv = pd.DataFrame({'Rank': range(1, len(top_hashtags)+1),
                                  'Hashtag': ['#'+h for h in top_hashtags],
                                  'Frekuensi': [hashtag_freq[h] for h in top_hashtags]})
            st.dataframe(df_hv, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada hashtag ditemukan di dataset.")

    st.markdown("---")

    # ── STEP 4: Forecasting ──────────────────────────────────────
    st.header("Forecasting LightGBM + Optuna")
    st.info(f"Forecast **{FORECAST_DAYS} hari** ke depan | Optuna **{N_TRIALS} trials** per item | Test size **{int(TEST_SIZE*100)}%**")

    SAVE_PATH = "models/forecast_results.pkl"

    # --- Load hasil tersimpan (tanpa run ulang) ---
    col_run, col_load = st.columns([2, 1])

    with col_load:
        if os.path.exists(SAVE_PATH):
            if st.button("📂 Load Hasil Tersimpan", use_container_width=True):
                with open(SAVE_PATH, "rb") as f:
                    saved = pickle.load(f)
                st.session_state['topic_results']   = saved['topic_results']
                st.session_state['word_results']    = saved['word_results']
                st.session_state['hashtag_results'] = saved['hashtag_results']
                saved_time = saved.get('saved_at', '-')
                st.success(f"✅ Hasil dimuat! (Disimpan: {saved_time})")
        else:
            st.info("Belum ada hasil tersimpan.")

    with col_run:
        run_forecast = st.button("Jalankan Forecasting (semua komponen)", type="primary", use_container_width=True)

    if run_forecast:

        topic_results   = {}
        word_results    = {}
        hashtag_results = {}

        # --- Topik ---
        st.subheader("Forecasting Topik...")
        prog_t = st.progress(0, text="Memproses topik...")
        for i, topic in enumerate(top_topics):
            prog_t.progress((i+1)/len(top_topics), text=f"Topik [{i+1}/{len(top_topics)}]: {topic[:50]}")
            sub_df = df_no_outlier[df_no_outlier['topic_name'] == topic][['tanggal']].copy()
            ts     = prepare_time_series(sub_df, 'tanggal')
            res    = forecast_with_lgbm(ts, topic)
            if res:
                topic_results[topic] = res
        prog_t.empty()

        # --- Kata ---
        st.subheader("Forecasting Kata...")
        prog_w = st.progress(0, text="Memproses kata...")
        for i, word in enumerate(top_words):
            prog_w.progress((i+1)/len(top_words), text=f"Kata [{i+1}/{len(top_words)}]: {word}")
            sub_df = words_df[words_df['kata'] == word][['tanggal']].copy()
            ts     = prepare_time_series(sub_df, 'tanggal')
            res    = forecast_with_lgbm(ts, word)
            if res:
                word_results[word] = res
        prog_w.empty()

        # --- Hashtag ---
        if top_hashtags:
            st.subheader("Forecasting Hashtag...")
            prog_h = st.progress(0, text="Memproses hashtag...")
            for i, ht in enumerate(top_hashtags):
                prog_h.progress((i+1)/len(top_hashtags), text=f"Hashtag [{i+1}/{len(top_hashtags)}]: #{ht}")
                sub_df = hashtags_df[hashtags_df['hashtag'] == ht][['tanggal']].copy()
                ts     = prepare_time_series(sub_df, 'tanggal')
                res    = forecast_with_lgbm(ts, f"#{ht}")
                if res:
                    hashtag_results[ht] = res
            prog_h.empty()

        st.session_state['topic_results']   = topic_results
        st.session_state['word_results']    = word_results
        st.session_state['hashtag_results'] = hashtag_results

        # --- Auto-save hasil ---
        os.makedirs("models", exist_ok=True)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump({
                'topic_results':   topic_results,
                'word_results':    word_results,
                'hashtag_results': hashtag_results,
                'saved_at':        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f)

        st.success(f"✅ Forecasting selesai dan otomatis tersimpan ke `{SAVE_PATH}`!")

    # ── STEP 5: Tampilkan Hasil ──────────────────────────────────
    if 'topic_results' not in st.session_state:
        st.warning("Klik tombol di atas untuk menjalankan forecasting.")
        return

    topic_results   = st.session_state['topic_results']
    word_results    = st.session_state['word_results']
    hashtag_results = st.session_state['hashtag_results']

    st.markdown("---")
    st.header("Hasil Forecasting")

    res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
        "Topik", "Kata", "Hashtag", "Ringkasan Evaluasi"
    ])

    # ---- TAB TOPIK ----
    with res_tab1:
        st.subheader(f"Forecast Frekuensi Top {TOP_N_ITEMS} Topik")
        if not topic_results:
            st.warning("Tidak ada hasil forecasting topik.")
        for item_name, res in topic_results.items():
            with st.expander(f"{item_name}", expanded=False):
                st.plotly_chart(
                    make_forecast_chart(res, f"Forecast: {item_name}"),
                    use_container_width=True
                )
                make_metrics_cols(res)


    # ---- TAB KATA ----
    with res_tab2:
        st.subheader(f"Forecast Frekuensi Top {TOP_N_ITEMS} Kata")
        if not word_results:
            st.warning("Tidak ada hasil forecasting kata.")
        for item_name, res in word_results.items():
            with st.expander(f"{item_name}", expanded=False):
                st.plotly_chart(
                    make_forecast_chart(res, f"Forecast: {item_name}"),
                    use_container_width=True
                )
                make_metrics_cols(res)

    # ---- TAB HASHTAG ----
    with res_tab3:
        st.subheader(f"Forecast Frekuensi Top {TOP_N_ITEMS} Hashtag")
        if not hashtag_results:
            st.warning("Tidak ada hasil forecasting hashtag.")
        for item_name, res in hashtag_results.items():
            with st.expander(f"{item_name}", expanded=False):
                st.plotly_chart(
                    make_forecast_chart(res, f"Forecast: {item_name}"),
                    use_container_width=True
                )
                make_metrics_cols(res)

    # ---- TAB RINGKASAN EVALUASI ----
    with res_tab4:
        st.subheader("Ringkasan Evaluasi Model")

        def build_summary(results_dict):
            rows = []
            for name, res in results_dict.items():
                m = res['test_metrics']
                rows.append({
                    'Item':           res['item_name'],
                    'Historical Avg': round(res['full_ts']['y'].mean(), 2),
                    'Historical Max': int(res['full_ts']['y'].max()),
                    'Forecast Avg':   round(res['forecast_df']['y_pred'].mean(), 2),
                    'Forecast Max':   round(res['forecast_df']['y_pred'].max(), 2),
                    'MAE':            round(m['MAE'], 4),
                    'RMSE':           round(m['RMSE'], 4),
                    'RMSSE':          round(m['RMSSE'], 4),
                    'Best RMSE (Optuna)': round(res['best_rmse'], 4),
                })
            return pd.DataFrame(rows)

        for label, results in [
            ("Topik",   topic_results),
            ("Kata",    word_results),
            ("Hashtag", hashtag_results),
        ]:
            if results:
                st.markdown(f"#### {label}")
                summary_df = build_summary(results)
                st.dataframe(
                    summary_df.style
                        .highlight_min(subset=['MAE','RMSE','RMSSE'], color='#d4edda')
                        .highlight_max(subset=['MAE','RMSE','RMSSE'], color='#f8d7da'),
                    use_container_width=True, hide_index=True
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Rata-rata MAE ({label})",   f"{summary_df['MAE'].mean():.4f}")
                c2.metric(f"Rata-rata RMSE ({label})",  f"{summary_df['RMSE'].mean():.4f}")
                c3.metric(f"Rata-rata RMSSE ({label})", f"{summary_df['RMSSE'].mean():.4f}")
                st.markdown("")