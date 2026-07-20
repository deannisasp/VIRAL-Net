import os
# CRITICAL: Set environment variables BEFORE importing transformers
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force transformers to use PyTorch backend only
os.environ['TRANSFORMERS_OFFLINE'] = '0'
# Disable TensorFlow
import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import transformers AFTER setting environment variables
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import random
import io

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show():
    st.title("Model Klasifikasi - Prediksi Viralitas Tweet")
    st.markdown("---")
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Tab untuk berbagai bagian
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Upload & Pelabelan",
        "Training Model", 
        "Evaluasi & Perbandingan",
        "Feature Importance",
        "Prediksi Tweet Baru"
    ])
    
    with tab1:
        show_data_upload_and_labeling()
    
    with tab2:
        show_model_training()
    
    with tab3:
        show_model_evaluation()
    
    with tab4:
        show_feature_importance()
    
    with tab5:
        show_prediction_interface()

def show_data_upload_and_labeling():
    """Upload data dan proses pelabelan dengan GMM"""
    st.header("Upload Data & Pelabelan GMM")
    
    st.markdown("""
    ### Langkah 1: Upload Data Bersih
    Upload file Excel yang berisi data tweet yang sudah di-preprocessing.
    
    **Kolom yang diperlukan:**
    - `final_tweet`: Teks tweet yang sudah dibersihkan
    - `followers`: Jumlah followers
    - `following`: Jumlah following
    - `like`: Jumlah like
    - `retweet`: Jumlah retweet
    - `verified_status`: Status verifikasi
    - `reply`: Jumlah reply
    - `quote`: Jumlah quote
    - `tweet_len`: Panjang tweet
    - `hashtags_count`: Jumlah hashtag
    - `mentions_count`: Jumlah mention
    """)
    
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Data berhasil dimuat! Total: {len(df)} baris")
            
            # Show data preview
            st.subheader("Preview Data")
            st.dataframe(df.head(10))
            
            # Proses pelabelan
            if st.button("Mulai Pelabelan dengan GMM", type="primary"):
                with st.spinner("Memproses pelabelan dengan Gaussian Mixture Model..."):
                    # Perform GMM labeling
                    df_labeled = perform_gmm_labeling(df)
                    
                    # Store in session state
                    st.session_state.df = df_labeled
                    st.session_state.data_processed = True
                    
                    st.success("✅ Pelabelan selesai!")
                    st.rerun()
            
            # Show labeling results if already processed
            if st.session_state.data_processed:
                st.markdown("---")
                st.subheader("Hasil Pelabelan")
                
                df_labeled = st.session_state.df
                
                # Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Data", len(df_labeled))
                    label_dist = df_labeled['label_gmm'].value_counts()
                    st.write("**Distribusi Label:**")
                    for label, count in label_dist.items():
                        st.write(f"- {label}: {count} ({count/len(df_labeled)*100:.1f}%)")
                
                with col2:
                    # Pie chart
                    fig = px.pie(
                        values=label_dist.values,
                        names=label_dist.index,
                        title="Distribusi Label GMM"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster visualization
                # st.subheader("Visualisasi Cluster (PCA 2D)")
                
                cols = ['followers', 'following', 'like', 'retweet', 'verified_status', 'reply', 'quote']
                X = np.log1p(df_labeled[cols].values)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                    c=df_labeled['cluster'], 
                                    cmap='viridis', s=40, alpha=0.6)
                handles, _ = scatter.legend_elements()
                ax.legend(handles, ['Tidak Viral', 'Viral'])
                # plt.colorbar(scatter, ax=ax, label='Cluster')
                # ax.set_xlabel('PCA Component 1')
                # ax.set_ylabel('PCA Component 2')
                # ax.set_title('Visualisasi Cluster GMM (PCA 2D)')
                # st.pyplot(fig)
                
                # Rata-rata fitur per cluster
                st.subheader("Rata-rata Fitur per Cluster")
                summary = df_labeled.groupby('label_gmm')[cols].mean().round(2)
                st.dataframe(summary, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    else:
        st.info("Silakan upload file Excel untuk memulai")

def perform_gmm_labeling(df):
    """Perform GMM clustering for labeling"""
    cols = ['followers', 'following', 'like', 'retweet', 'verified_status', 'reply', 'quote']
    
    # Validate and clip values
    df[cols] = df[cols].clip(lower=0)
    
    # Log transform
    X = np.log1p(df[cols].values)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # GMM
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='diag',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    df['cluster'] = gmm.fit_predict(X_scaled)
    
    # Determine viral cluster
    cluster_means = gmm.means_.mean(axis=1)
    viral_cluster = np.argmax(cluster_means)
    df['cluster'] = df['cluster'].map({0: 1, 1: 0})
    
    df['label_gmm'] = df['cluster'].apply(
        lambda c: 'tidak viral' if c == viral_cluster else 'viral'
    )
    
    return df

def show_model_training():
    """Train the main model"""
    st.header("Training Model Utama")
    
    if not st.session_state.data_processed:
        st.warning("⚠️ Silakan upload dan proses data terlebih dahulu di tab 'Upload & Pelabelan'")
        return
    
    st.markdown("""
    ### Model Architecture
    - **Text Embedding**: IndoBERTweet (768-dim)
    - **Sentiment Analysis**: RoBERTa Sentiment Head (3-class)
    - **Numeric Features**: 6 features (followers, following, verified_status, tweet_len, hashtags_count, mentions_count)
    - **Classifier**: MLP (Linear(777→512) → BN → ReLU → Dropout(0.3) → Linear(512→2))
    - **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
    - **Loss**: CrossEntropyLoss dengan class weights (balanced)
    - **Early Stopping**: Patience=3, metric=Macro F1
    """)
    
    # Training parameters
    st.subheader("Parameter Training")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_epochs = st.number_input("Jumlah Epoch", min_value=5, max_value=30, value=15)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=8, max_value=64, value=32)
    with col3:
        learning_rate = st.selectbox("Learning Rate", [5e-5, 1e-3, 1e-4, 1e-5], index=0)
    
    if st.button("Mulai Training", type="primary"):
        with st.spinner("Training model... Ini mungkin memakan waktu beberapa menit"):
            try:
                # Train the model
                results = train_main_model(
                    st.session_state.df,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
                # Store results in session state
                st.session_state.training_results = results
                st.session_state.model_trained = True
                
                st.success("✅ Training selesai!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.exception(e)
    
    # Show training history if available
    if st.session_state.model_trained:
        st.markdown("---")
        st.subheader("Training History")
        
        results = st.session_state.training_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curve
            epochs = list(range(1, len(results['train_losses']) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=results['train_losses'], 
                                    mode='lines+markers', name='Train Loss'))
            fig.add_trace(go.Scatter(x=epochs, y=results['val_losses'], 
                                    mode='lines+markers', name='Val Loss'))
            fig.update_layout(
                title='Loss over Epochs',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Val metrics
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=results['val_f1s'], 
                                    mode='lines+markers', name='Val F1'))
            fig.add_trace(go.Scatter(x=epochs, y=results['val_accs'], 
                                    mode='lines+markers', name='Val Accuracy'))
            fig.update_layout(
                title='Validation Metrics over Epochs',
                xaxis_title='Epoch',
                yaxis_title='Score',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Training stopped at epoch {len(results['train_losses'])} "
                f"(Early stopping patience: 3 epochs)")

def train_main_model(df, num_epochs=15, batch_size=32, learning_rate=1e-4):
    """Train the main viral classification model"""
    
    # Prepare data
    required_cols = ['final_tweet', 'followers', 'following', 'verified_status',
                    'tweet_len', 'hashtags_count', 'mentions_count', 'label_gmm']
    
    df = df.dropna(subset=['final_tweet', 'label_gmm']).reset_index(drop=True)
    
    # Encode labels
    if df['label_gmm'].dtype == object:
        le = LabelEncoder()
        df['label_gmm'] = le.fit_transform(df['label_gmm'].astype(str))
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, 
                                         stratify=df['label_gmm'])
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=SEED, 
                                        stratify=train_df['label_gmm'])
    
    # Numeric features scaling
    numeric_cols = ['followers', 'following', 'verified_status', 'tweet_len', 
                   'hashtags_count', 'mentions_count']
    
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols].values)
    train_num = scaler.transform(train_df[numeric_cols].values)
    val_num = scaler.transform(val_df[numeric_cols].values)
    test_num = scaler.transform(test_df[numeric_cols].values)
    
    # Load models
    bert_name = "indolem/indobertweet-base-uncased"
    roberta_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModel.from_pretrained(bert_name).to(device)
    roberta_full = AutoModelForSequenceClassification.from_pretrained(roberta_name)
    
    # Create embedders
    bert_embedder = IndoBERTweetEmbedder(bert_model).to(device)
    roberta_head = RobertaSentimentHeadFromEmbed(roberta_full).to(device)
    
    # Compute embeddings
    bert_train = compute_bert_embeddings(train_df['final_tweet'].tolist(), 
                                        bert_tokenizer, bert_embedder)
    bert_val = compute_bert_embeddings(val_df['final_tweet'].tolist(), 
                                      bert_tokenizer, bert_embedder)
    bert_test = compute_bert_embeddings(test_df['final_tweet'].tolist(), 
                                       bert_tokenizer, bert_embedder)
    
    # Compute sentiment
    sent_train_probs = compute_sentiment_from_bert_embeds(bert_train, roberta_head)
    sent_val_probs = compute_sentiment_from_bert_embeds(bert_val, roberta_head)
    sent_test_probs = compute_sentiment_from_bert_embeds(bert_test, roberta_head)
    
    # Prepare tensors
    y_train = torch.tensor(train_df['label_gmm'].values, dtype=torch.long)
    y_val = torch.tensor(val_df['label_gmm'].values, dtype=torch.long)
    y_test = torch.tensor(test_df['label_gmm'].values, dtype=torch.long)

    X_train_num = torch.tensor(train_num, dtype=torch.float)
    X_val_num = torch.tensor(val_num, dtype=torch.float)
    X_test_num = torch.tensor(test_num, dtype=torch.float)

    # Target regresi: log1p transform
    reg_cols = ['like', 'retweet', 'quote', 'reply']
    for col in reg_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)

    y_train_reg = torch.tensor(np.log1p(train_df[reg_cols].values), dtype=torch.float)
    y_val_reg   = torch.tensor(np.log1p(val_df[reg_cols].values),   dtype=torch.float)
    y_test_reg  = torch.tensor(np.log1p(test_df[reg_cols].values),  dtype=torch.float)

    # Create dataloaders (sertakan y_reg)
    train_dataset = TensorDataset(bert_train, sent_train_probs, X_train_num, y_train, y_train_reg)
    val_dataset   = TensorDataset(bert_val,   sent_val_probs,   X_val_num,   y_val,   y_val_reg)
    test_dataset  = TensorDataset(bert_test,  sent_test_probs,  X_test_num,  y_test,  y_test_reg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = len(np.unique(df['label_gmm'].values))
    mlp = ViralClassifierMLP(
        bert_dim=bert_train.shape[1],
        sent_dim=sent_train_probs.shape[1],
        numeric_dim=X_train_num.shape[1],
        hidden_dim=512,
        num_classes=num_classes
    ).to(device)
    
    # Class weights
    classes = np.unique(y_train.numpy())
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Optimizer and loss
    optimizer     = torch.optim.AdamW(mlp.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_reg = nn.MSELoss()

    # Bobot loss multi-task
    ALPHA = 1.0   # bobot klasifikasi
    BETA  = 0.25  # bobot regresi

    # Early stopping
    early_stopper = EarlyStopping(patience=5, min_delta=1e-4, mode="max")

    # Training loop
    train_losses = []
    val_losses = []
    val_f1s = []
    val_accs = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(1, num_epochs + 1):
        mlp.train()
        running_loss = 0.0

        for bert_b, sent_b, num_b, y_b, y_reg_b in train_loader:
            bert_b    = bert_b.to(device)
            sent_b    = sent_b.to(device)
            num_b     = num_b.to(device)
            y_b       = y_b.to(device)
            y_reg_b   = y_reg_b.to(device)

            optimizer.zero_grad()
            cls_logits, like_p, retweet_p, quote_p, reply_p = mlp(bert_b, sent_b, num_b)

            loss_cls     = criterion_cls(cls_logits, y_b)
            loss_like    = criterion_reg(like_p.squeeze(), y_reg_b[:, 0])
            loss_retweet = criterion_reg(retweet_p.squeeze(), y_reg_b[:, 1])
            loss_quote   = criterion_reg(quote_p.squeeze(), y_reg_b[:, 2])
            loss_reply   = criterion_reg(reply_p.squeeze(), y_reg_b[:, 3])
            loss = ALPHA * loss_cls + BETA * (loss_like + loss_retweet + loss_quote + loss_reply)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bert_b.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        mlp.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for bert_b, sent_b, num_b, y_b, y_reg_b in val_loader:
                bert_b    = bert_b.to(device)
                sent_b    = sent_b.to(device)
                num_b     = num_b.to(device)
                y_b       = y_b.to(device)
                y_reg_b   = y_reg_b.to(device)

                cls_logits, like_p, retweet_p, quote_p, reply_p = mlp(bert_b, sent_b, num_b)

                loss_cls     = criterion_cls(cls_logits, y_b)
                loss_like    = criterion_reg(like_p.squeeze(), y_reg_b[:, 0])
                loss_retweet = criterion_reg(retweet_p.squeeze(), y_reg_b[:, 1])
                loss_quote   = criterion_reg(quote_p.squeeze(), y_reg_b[:, 2])
                loss_reply   = criterion_reg(reply_p.squeeze(), y_reg_b[:, 3])
                loss = ALPHA * loss_cls + BETA * (loss_like + loss_retweet + loss_quote + loss_reply)

                val_running_loss += loss.item() * bert_b.size(0)

                preds = torch.argmax(torch.softmax(cls_logits, dim=1), dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_b.cpu().numpy())
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = balanced_accuracy_score(all_labels, all_preds)
        
        val_f1s.append(f1)
        val_accs.append(acc)
        
        # Update progress
        progress_bar.progress(epoch / num_epochs)
        status_text.text(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, "
                        f"Val Loss: {epoch_val_loss:.4f}, Val F1: {f1:.4f}")
        
        # Early stopping
        early_stopper(f1, mlp)
        if early_stopper.early_stop:
            mlp.load_state_dict(early_stopper.best_state)
            break
    
    # Final evaluation on test set
    mlp.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_like, all_retweet, all_quote, all_reply = [], [], [], []
    all_like_true, all_retweet_true, all_quote_true, all_reply_true = [], [], [], []

    with torch.no_grad():
        for bert_b, sent_b, num_b, y_b, y_reg_b in test_loader:
            bert_b  = bert_b.to(device)
            sent_b  = sent_b.to(device)
            num_b   = num_b.to(device)

            cls_logits, like_p, retweet_p, quote_p, reply_p = mlp(bert_b, sent_b, num_b)
            probs = torch.softmax(cls_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_b.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # Regresi
            all_like.append(np.expm1(like_p.squeeze().cpu().numpy()))
            all_retweet.append(np.expm1(retweet_p.squeeze().cpu().numpy()))
            all_quote.append(np.expm1(quote_p.squeeze().cpu().numpy()))
            all_reply.append(np.expm1(reply_p.squeeze().cpu().numpy()))

            all_like_true.append(np.expm1(y_reg_b[:, 0].numpy()))
            all_retweet_true.append(np.expm1(y_reg_b[:, 1].numpy()))
            all_quote_true.append(np.expm1(y_reg_b[:, 2].numpy()))
            all_reply_true.append(np.expm1(y_reg_b[:, 3].numpy()))
    
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    pred_like    = np.concatenate(all_like)
    pred_retweet = np.concatenate(all_retweet)
    pred_quote   = np.concatenate(all_quote)
    pred_reply   = np.concatenate(all_reply)

    true_like    = np.concatenate(all_like_true)
    true_retweet = np.concatenate(all_retweet_true)
    true_quote   = np.concatenate(all_quote_true)
    true_reply   = np.concatenate(all_reply_true)

    # Compute regression metrics (log-space)
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    reg_metrics = {}
    for name, pred, true in [
        ('like',    pred_like,    true_like),
        ('retweet', pred_retweet, true_retweet),
        ('quote',   pred_quote,   true_quote),
        ('reply',   pred_reply,   true_reply),
    ]:
        log_pred = np.log1p(pred)
        log_true = np.log1p(true)
        reg_metrics[name] = {
            'log_mae':  mean_absolute_error(log_true, log_pred),
            'log_rmse': np.sqrt(mean_squared_error(log_true, log_pred)),
            'med_ae':   float(np.median(np.abs(log_true - log_pred))),
        }
    
    # Compute final metrics
    test_metrics = {
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds)
    }
    
    if num_classes == 2:
        try:
            test_metrics['roc_auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            test_metrics['roc_auc'] = None
    
    # Store everything for later use
    results = {
        'model': mlp,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1s': val_f1s,
        'val_accs': val_accs,
        'test_metrics': test_metrics,
        'reg_metrics': reg_metrics,
        'test_predictions': all_preds,
        'test_labels': all_labels,
        'test_probs': all_probs,
        'bert_embedder': bert_embedder,
        'roberta_head': roberta_head,
        'bert_tokenizer': bert_tokenizer,
        'scaler': scaler,
        'numeric_cols': numeric_cols,
        'num_classes': num_classes,
        'test_loader': test_loader,
        'bert_test': bert_test,
        'sent_test_probs': sent_test_probs,
        'X_test_num': X_test_num,
        'y_test': y_test,
        'y_test_reg': y_test_reg,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'bert_train': bert_train,
        'bert_val': bert_val,
        'sent_train_probs': sent_train_probs,
        'sent_val_probs': sent_val_probs,
        'X_train_num': X_train_num,
        'X_val_num': X_val_num,
        'y_train': y_train,
        'y_val': y_val,
        'y_train_reg': y_train_reg,
        'y_val_reg': y_val_reg,
        'class_weights': class_weights,
        'batch_size': batch_size
    }
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def show_model_evaluation():
    """Show model evaluation results"""
    st.header("Evaluasi Model & Perbandingan")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Silakan train model terlebih dahulu di tab 'Training Model'")
        return
    
    results = st.session_state.training_results
    
    # Test metrics
    st.subheader("Test Set Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = results['test_metrics']
    
    with col1:
        st.metric("F1 Score", f"{metrics['f1']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("Balanced Accuracy", f"{metrics['balanced_acc']:.4f}")
    
    if metrics.get('roc_auc'):
        st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        cm = confusion_matrix(results['test_labels'], results['test_predictions'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Viral', 'Viral'],
                   yticklabels=['Not Viral', 'Viral'],
                   ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix - Test Set')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Interpretasi:")
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **True Negative**: {tn}
        - **False Positive**: {fp}
        - **False Negative**: {fn}
        - **True Positive**: {tp}
        
        **Accuracy**: {(tn+tp)/(tn+fp+fn+tp):.2%}
        """)
    
    # ROC Curve
    if results['num_classes'] == 2 and metrics.get('roc_auc'):
        st.subheader("ROC Curve")
        
        fpr, tpr, _ = roc_curve(results['test_labels'], results['test_probs'][:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Random',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Regression metrics
    st.markdown("---")
    st.subheader("Evaluasi Regression Head (Prediksi Engagement)")

    st.caption(
        "Metrik dihitung dalam log-space karena distribusi engagement sangat skewed (mayoritas 0). "
        "Log-MAE < 1.0 sudah tergolong baik untuk data seperti ini."
    )

    reg_metrics = results.get('reg_metrics', {})
    if reg_metrics:
        reg_rows = []
        for name, m in reg_metrics.items():
            reg_rows.append({
                'Engagement':    name.capitalize(),
                'Log-MAE':       round(m['log_mae'],  4),
                'Log-RMSE':      round(m['log_rmse'], 4),
                'Median Log-AE': round(m['med_ae'],   4),
            })
        reg_df = pd.DataFrame(reg_rows)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
    else:
        st.info("Regression metrics tidak tersedia.")

    # Baseline comparison (dinonaktifkan)
    # st.markdown("---")
    # st.subheader("Perbandingan dengan Baseline Models")
    # ...

def train_baseline_models(results):
    """Train baseline models for comparison"""
    # Prepare features
    X_train_all = np.concatenate([
        results['bert_train'].numpy(), 
        results['sent_train_probs'].numpy(), 
        results['X_train_num'].numpy()
    ], axis=1)
    
    X_test_all = np.concatenate([
        results['bert_test'].numpy(), 
        results['sent_test_probs'].numpy(), 
        results['X_test_num'].numpy()
    ], axis=1)
    
    y_train = results['y_train'].numpy()
    y_test = results['y_test'].numpy()
    
    # Train models
    models = {
        "Logistic Regression": LogisticRegression(solver="newton-cg", class_weight="balanced", max_iter=500),
        "SGD Classifier": SGDClassifier(class_weight="balanced", random_state=SEED),
        "Decision Tree": tree.DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED, class_weight="balanced", n_estimators=100),
    }
    
    baseline_results = {}
    
    for name, model in models.items():
        model.fit(X_train_all, y_train)
        y_pred = model.predict(X_test_all)
        
        baseline_results[name] = {
            'f1': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred)
        }
    
    return baseline_results

def show_feature_importance():
    """Show feature importance through ablation study"""
    st.header("Feature Importance (Ablation Study)")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Silakan train model terlebih dahulu di tab 'Training Model'")
        return
    
    st.markdown("""
    ### Ablation Study
    Analisis kontribusi setiap fitur numerik dengan menghilangkan satu fitur pada satu waktu dan
    mengamati dampaknya terhadap performa model.
    """)
    
    if st.button("Mulai Ablation Study", type="primary"):
        with st.spinner("Menjalankan Ablation Study... Ini mungkin memakan waktu beberapa menit"):
            results = st.session_state.training_results
            ablation_results = run_ablation_study(results)
            st.session_state.ablation_results = ablation_results
            st.success("✅ Ablation study completed!")
            st.rerun()
    
    if 'ablation_results' in st.session_state:
        ablation_results = st.session_state.ablation_results
        baseline_f1 = ablation_results['BASELINE_FULL_MODEL']['f1']
        
        # Feature importance table
        st.subheader("Feature Importance Analysis")
        
        feature_impact = {}
        numeric_cols = st.session_state.training_results['numeric_cols']
        
        for feature in numeric_cols:
            key = f'WITHOUT_{feature}'
            if key in ablation_results:
                delta_f1 = baseline_f1 - ablation_results[key]['f1']
                feature_impact[feature] = delta_f1
        
        sorted_impact = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
        
        impact_data = {
            'Feature': [f[0] for f in sorted_impact],
            'F1 Drop': [f[1] for f in sorted_impact],
            'Importance': ['Low' if f[1] < 0 else 'High' if f[1] > 0.01 else 'Medium' 
                          for f in sorted_impact]
        }
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)
        
        # Bar chart
        fig = go.Figure(go.Bar(
            x=impact_df['F1 Drop'],
            y=impact_df['Feature'],
            orientation='h',
            marker=dict(
                color=impact_df['F1 Drop'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title='Feature Importance - F1 Score Drop When Removed',
            xaxis_title='F1 Score Drop',
            yaxis_title='Feature',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Full results table
        st.subheader("Full Ablation Results")
        
        results_data = []
        for key, metrics in ablation_results.items():
            results_data.append({
                'Configuration': key,
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Balanced Acc': metrics['balanced_acc']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('F1', ascending=True)
        
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['F1', 'Precision', 'Recall', 'Balanced Acc']),
                    use_container_width=True)

def run_ablation_study(results):
    """Run ablation study by removing features one at a time"""
    baseline_metrics = results['test_metrics']
    ablation_results = {'BASELINE_FULL_MODEL': baseline_metrics}
    
    numeric_cols = results['numeric_cols']
    batch_size = results['batch_size']
    
    # Get original data
    train_num = results['X_train_num'].numpy()
    val_num = results['X_val_num'].numpy()
    test_num = results['X_test_num'].numpy()
    
    # Remove one feature at a time
    for i, feature_to_remove in enumerate(numeric_cols):
        remaining_features = [f for f in numeric_cols if f != feature_to_remove]
        remaining_indices = [numeric_cols.index(f) for f in remaining_features]
        
        X_train_ablation = torch.tensor(train_num[:, remaining_indices], dtype=torch.float)
        X_val_ablation = torch.tensor(val_num[:, remaining_indices], dtype=torch.float)
        X_test_ablation = torch.tensor(test_num[:, remaining_indices], dtype=torch.float)
        
        train_loader_abl = DataLoader(
            TensorDataset(results['bert_train'], results['sent_train_probs'], 
                         X_train_ablation, results['y_train']),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader_abl = DataLoader(
            TensorDataset(results['bert_val'], results['sent_val_probs'], 
                         X_val_ablation, results['y_val']),
            batch_size=batch_size, shuffle=False
        )
        test_loader_abl = DataLoader(
            TensorDataset(results['bert_test'], results['sent_test_probs'], 
                         X_test_ablation, results['y_test']),
            batch_size=batch_size, shuffle=False
        )
        
        metrics, _ = train_and_evaluate_ablation(
            train_loader_abl, val_loader_abl, test_loader_abl,
            bert_dim=results['bert_train'].shape[1],
            sent_dim=results['sent_train_probs'].shape[1],
            numeric_dim=len(remaining_features),
            num_classes=results['num_classes'],
            class_weights=results['class_weights']
        )
        
        ablation_results[f'WITHOUT_{feature_to_remove}'] = metrics
    
    # Extreme ablation: no numeric features
    X_train_empty = torch.zeros((len(results['y_train']), 0))
    X_val_empty = torch.zeros((len(results['y_val']), 0))
    X_test_empty = torch.zeros((len(results['y_test']), 0))
    
    train_loader_no_num = DataLoader(
        TensorDataset(results['bert_train'], results['sent_train_probs'], 
                     X_train_empty, results['y_train']),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader_no_num = DataLoader(
        TensorDataset(results['bert_val'], results['sent_val_probs'], 
                     X_val_empty, results['y_val']),
        batch_size=batch_size, shuffle=False
    )
    test_loader_no_num = DataLoader(
        TensorDataset(results['bert_test'], results['sent_test_probs'], 
                     X_test_empty, results['y_test']),
        batch_size=batch_size, shuffle=False
    )
    
    metrics_no_num, _ = train_and_evaluate_ablation(
        train_loader_no_num, val_loader_no_num, test_loader_no_num,
        bert_dim=results['bert_train'].shape[1],
        sent_dim=results['sent_train_probs'].shape[1],
        numeric_dim=0,
        num_classes=results['num_classes'],
        class_weights=results['class_weights']
    )
    
    ablation_results['NO_NUMERIC_FEATURES'] = metrics_no_num
    
    return ablation_results

def train_and_evaluate_ablation(train_loader, val_loader, test_loader,
                                bert_dim, sent_dim, numeric_dim,
                                num_classes, class_weights,
                                num_epochs=15, patience=3, lr=1e-4):
    """Train and evaluate model for ablation study"""
    mlp = ViralClassifierMLPFlexible(
        bert_dim=bert_dim,
        sent_dim=sent_dim,
        numeric_dim=numeric_dim,
        hidden_dim=512,
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")
    
    for epoch in range(num_epochs):
        mlp.train()
        for bert_b, sent_b, num_b, y_b in train_loader:
            bert_b, sent_b, num_b, y_b = (
                bert_b.to(device),
                sent_b.to(device),
                num_b.to(device),
                y_b.to(device),
            )
            optimizer.zero_grad()
            loss = criterion(mlp(bert_b, sent_b, num_b), y_b)
            loss.backward()
            optimizer.step()
        
        mlp.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bert_b, sent_b, num_b, y_b in val_loader:
                logits = mlp(
                    bert_b.to(device),
                    sent_b.to(device),
                    num_b.to(device)
                )
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_b.numpy())
        
        f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro")
        early_stopper(f1, mlp)
        if early_stopper.early_stop:
            mlp.load_state_dict(early_stopper.best_state)
            break
    
    # Test
    mlp.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bert_b, sent_b, num_b, y_b in test_loader:
            logits = mlp(
                bert_b.to(device),
                sent_b.to(device),
                num_b.to(device)
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_b.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return {
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
        "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
    }, mlp

# def show_prediction_interface():
#     """Interface for predicting new tweets"""
#     st.header("Prediksi Viralitas Tweet Baru")
    
#     if not st.session_state.model_trained:
#         st.warning("⚠️ Silakan train model terlebih dahulu di tab 'Training Model'")
#         return
    
#     st.markdown("""
#     ### Masukkan Detail Tweet
#     Sistem akan memprediksi probabilitas tweet Anda menjadi viral berdasarkan model yang sudah dilatih.
#     """)
    
#     st.markdown("---")
    
#     # Form input
#     with st.form("prediction_form"):
#         st.subheader("Konten Tweet")
#         tweet_text = st.text_area(
#             "Masukkan teks tweet",
#             height=150,
#             max_chars=280,
#             placeholder="Contoh: Jangan lewatkan promo spesial hari ini! Diskon hingga 70% untuk semua produk favorit kamu 🎉"
#         )
        
#         st.markdown("---")
#         st.subheader("Informasi Akun")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             followers = st.number_input(
#                 "Jumlah Followers",
#                 min_value=0,
#                 max_value=1000000000,
#                 value=1000,
#                 step=100
#             )
        
#         with col2:
#             following = st.number_input(
#                 "Jumlah Following",
#                 min_value=0,
#                 max_value=1000000000,
#                 value=500,
#                 step=10
#             )
        
#         with col3:
#             verified = st.selectbox(
#                 "Status Verifikasi",
#                 options=[
#                     ("Tidak Terverifikasi", 0),
#                     ("Terverifikasi", 1)
#                 ],
#                 format_func=lambda x: x[0]
#             )
        
#         st.markdown("---")
#         submit_button = st.form_submit_button("Prediksi Viralitas", type="primary")
    
#     # Process prediction
#     if submit_button:
#         if not tweet_text.strip():
#             st.error("⚠️ Teks tweet tidak boleh kosong!")
#         else:
#             with st.spinner("Memproses prediksi..."):
#                 # Extract features from tweet
#                 tweet_len = len(tweet_text)
#                 hashtags_count = tweet_text.count('#')
#                 mentions_count = tweet_text.count('@')
                
#                 # Predict
#                 # probability, label = predict_virality(
#                 #     tweet_text, followers, following, verified[1],
#                 #     tweet_len, hashtags_count, mentions_count,
#                 #     st.session_state.training_results
#                 # )

#                 label, probability, probs = predict_virality(
#                     tweet_text, followers, following, verified[1],
#                     tweet_len, hashtags_count, mentions_count,
#                     st.session_state.training_results
#                 )
                
#                 # Display results
#                 st.markdown("---")
#                 st.success("✅ Prediksi Selesai!")
#                 # Ambil probabilitas kelas viral & tidak viral
#                 prob_tidak_viral = probs[0]
#                 prob_viral = probs[1]
#                 st.write("Probabilitas Tidak Viral:", prob_tidak_viral)
#                 st.write("Probabilitas Viral:", prob_viral)
                
#                 col1, col2 = st.columns([1, 1])
                
#                 with col1:
#                     st.markdown("### Hasil Prediksi")
                    
#                     # Gauge chart
#                     fig = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=prob_viral * 100,
#                         domain={'x': [0, 1], 'y': [0, 1]},
#                         title={'text': "Probabilitas Viral (%)"},
#                         gauge={
#                             'axis': {'range': [None, 100]},
#                             'bar': {'color': "darkblue"},
#                             'steps': [
#                                 {'range': [0, 30], 'color': "lightgray"},
#                                 {'range': [30, 70], 'color': "gray"},
#                                 {'range': [70, 100], 'color': "lightgreen"}
#                             ],
#                             'threshold': {
#                                 'line': {'color': "red", 'width': 4},
#                                 'thickness': 0.75,
#                                 'value': 50
#                             }
#                         }
#                     ))
                    
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 with col2:
#                     st.markdown("### Interpretasi")
                    
#                     if label == "viral":
#                         st.success(f"""
#                         **Tweet Diprediksi: VIRAL**
                        
#                         Probabilitas Viral: **{prob_viral*100:.2f}%**
                        
#                         Tweet Anda memiliki peluang besar untuk viral!
#                         """)
#                     else:
#                         st.info(f"""
#                         **Tweet Diprediksi: TIDAK VIRAL**
                        
#                         Probabilitas Viral: **{prob_viral*100:.2f}%**
                        
#                         Tweet ini kemungkinan tidak akan viral.
#                         """)
                
#                 # Detail analisis
#                 st.markdown("---")
#                 st.markdown("### Detail Analisis")
                
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.metric("Panjang Tweet", f"{tweet_len} char")
                
#                 with col2:
#                     st.metric("Jumlah Hashtag", hashtags_count)
                
#                 with col3:
#                     st.metric("Jumlah Mention", mentions_count)
                
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.metric("Followers", f"{followers:,}")
                
#                 with col2:
#                     st.metric("Following", f"{following:,}")
                
#                 with col3:
#                     st.metric("Verified Status", verified[0])

def predict_virality(text, followers, following, verified_status,
                     tweet_len, hashtags_count, mentions_count, results):
    """Predict virality and engagement using trained multi-task model"""
    # Prepare numeric features
    numeric_features = np.array([[followers, following, verified_status,
                                  tweet_len, hashtags_count, mentions_count]])

    # Scale numeric features
    numeric_scaled = results['scaler'].transform(numeric_features)
    numeric_tensor = torch.tensor(numeric_scaled, dtype=torch.float).to(device)

    # Get BERT embedding
    bert_tokenizer = results['bert_tokenizer']
    bert_embedder  = results['bert_embedder']

    enc = bert_tokenizer([text], return_tensors="pt", padding=True,
                         truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()
           if k in ["input_ids", "attention_mask", "token_type_ids"]}

    with torch.no_grad():
        bert_embed = bert_embedder(**enc)

    # Get sentiment
    roberta_head = results['roberta_head']
    with torch.no_grad():
        sent_probs = roberta_head(bert_embed)

    # Predict (multi-task)
    model = results['model']
    model.eval()

    with torch.no_grad():
        cls_logits, like_p, retweet_p, quote_p, reply_p = model(bert_embed, sent_probs, numeric_tensor)
        probs     = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_cls  = np.argmax(probs)

        # Inverse log1p → nilai asli
        like_val    = float(np.expm1(like_p.cpu().numpy()[0, 0]))
        retweet_val = float(np.expm1(retweet_p.cpu().numpy()[0, 0]))
        quote_val   = float(np.expm1(quote_p.cpu().numpy()[0, 0]))
        reply_val   = float(np.expm1(reply_p.cpu().numpy()[0, 0]))

    label_map = {0: 'tidak viral', 1: 'viral'}
    return {
        'label'      : label_map[pred_cls],
        'confidence' : probs[pred_cls],
        'probs'      : probs,
        'like'       : round(like_val),
        'retweet'    : round(retweet_val),
        'quote'      : round(quote_val),
        'reply'      : round(reply_val),
    }

# Helper classes
class IndoBERTweetEmbedder(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                              token_type_ids=token_type_ids)
            cls = outputs.last_hidden_state[:, 0, :].detach()
        return cls

class RobertaSentimentHeadFromEmbed(nn.Module):
    def __init__(self, roberta_full_model):
        super().__init__()
        self.hidden_size = 768
        
        if hasattr(roberta_full_model, "classifier"):
            classifier = roberta_full_model.classifier
            if hasattr(classifier, "dense") and hasattr(classifier, "out_proj"):
                self.dense = nn.Linear(classifier.dense.in_features, classifier.dense.out_features)
                self.out_proj = nn.Linear(classifier.out_proj.in_features, classifier.out_proj.out_features)
                self.dense.load_state_dict(classifier.dense.state_dict())
                self.out_proj.load_state_dict(classifier.out_proj.state_dict())
            elif isinstance(classifier, nn.Linear):
                self.dense = None
                self.out_proj = nn.Linear(classifier.in_features, classifier.out_features)
                self.out_proj.load_state_dict(classifier.state_dict())
            else:
                raise ValueError("Unknown classifier structure")
        else:
            raise AttributeError("Model doesn't have 'classifier' attribute")
    
    def forward(self, bert_embeds):
        if hasattr(self, "dense") and self.dense is not None:
            x = torch.tanh(self.dense(bert_embeds))
            logits = self.out_proj(x)
        else:
            logits = self.out_proj(bert_embeds)
        probs = torch.softmax(logits, dim=1)
        return probs

class ViralClassifierMLP(nn.Module):
    """
    Multi-task MLP:
      - 1 head klasifikasi: viral vs tidak viral
      - 4 head regresi: prediksi like, retweet, quote, reply
    """
    def __init__(self, bert_dim=768, sent_dim=3, numeric_dim=6, hidden_dim=512, num_classes=2):
        super().__init__()
        input_dim = bert_dim + sent_dim + numeric_dim

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Head 1: Klasifikasi viral/tidak viral
        self.cls_head = nn.Linear(256, num_classes)

        # Head 2-5: Regresi jumlah interaksi (output log-scale, inverse saat prediksi)
        self.like_head    = nn.Linear(256, 1)
        self.retweet_head = nn.Linear(256, 1)
        self.quote_head   = nn.Linear(256, 1)
        self.reply_head   = nn.Linear(256, 1)

    def forward(self, bert_embed, sent_probs, numeric_feats):
        x = torch.cat([bert_embed, sent_probs, numeric_feats], dim=1)
        shared_out = self.shared(x)

        cls_logits   = self.cls_head(shared_out)
        like_pred    = F.softplus(self.like_head(shared_out))
        retweet_pred = F.softplus(self.retweet_head(shared_out))
        quote_pred   = F.softplus(self.quote_head(shared_out))
        reply_pred   = F.softplus(self.reply_head(shared_out))

        return cls_logits, like_pred, retweet_pred, quote_pred, reply_pred

class ViralClassifierMLPFlexible(nn.Module):
    def __init__(self, bert_dim=768, sent_dim=3, numeric_dim=6, hidden_dim=512, num_classes=2):
        super().__init__()
        self.numeric_dim = numeric_dim
        input_dim = bert_dim + sent_dim + numeric_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, bert_embed, sent_probs, numeric_feats):
        if self.numeric_dim > 0:
            x = torch.cat([bert_embed, sent_probs, numeric_feats], dim=1)
        else:
            x = torch.cat([bert_embed, sent_probs], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            return
        
        improvement = (score - self.best_score) if self.mode == "max" else (self.best_score - score)
        
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def compute_bert_embeddings(texts, tokenizer, embedder, batch_size=16, max_length=128):
    embed_list = []
    embed_device = next(embedder.parameters()).device
    embedder.eval()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, 
                       truncation=True, max_length=max_length)
        enc = {k: v.to(embed_device) for k, v in enc.items() 
               if k in ["input_ids", "attention_mask", "token_type_ids"]}
        with torch.no_grad():
            cls = embedder(**enc)
        embed_list.append(cls.cpu())
    
    return torch.cat(embed_list, dim=0)

def compute_sentiment_from_bert_embeds(bert_embeds, roberta_head, batch_size=32):
    roberta_head.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, bert_embeds.size(0), batch_size):
            b = bert_embeds[i:i+batch_size].to(device)
            probs = roberta_head(b)
            probs_list.append(probs.cpu())
    all_probs = torch.cat(probs_list, dim=0)
    return all_probs

import pickle

def load_saved_model(path="models/viral_classifier.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        save_data = pickle.load(f)
    return save_data

def build_model_from_saved(save_data):
    """Buat MLP multi-task dari saved data, dipanggil hanya saat prediksi"""
    mlp = ViralClassifierMLP(
        bert_dim    = save_data['bert_dim'],
        sent_dim    = save_data['sent_dim'],
        numeric_dim = len(save_data['numeric_cols']),
        hidden_dim  = 512,
        num_classes = save_data['num_classes']
    ).to(device)
    mlp.load_state_dict(save_data['model_state'])
    mlp.eval()
    save_data['model'] = mlp
    return save_data

def load_embedding_models():
    bert_name     = "indolem/indobertweet-base-uncased"
    roberta_name  = "w11wo/indonesian-roberta-base-sentiment-classifier"

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model_raw = AutoModel.from_pretrained(bert_name).to(device)
    roberta_full   = AutoModelForSequenceClassification.from_pretrained(roberta_name)

    bert_embedder = IndoBERTweetEmbedder(bert_model_raw).to(device)
    roberta_head  = RobertaSentimentHeadFromEmbed(roberta_full).to(device)

    return bert_tokenizer, bert_embedder, roberta_head


def show_prediction_interface():
    """Interface for predicting new tweets"""
    st.header("Prediksi Viralitas Tweet Baru")

    # ── Pilihan sumber model ─────────────────────────────────────
    st.subheader("Pilih Sumber Model")

    model_options = ["Model Bawaan"]
    if st.session_state.model_trained:
        model_options.append("Model Training Sendiri (sesi ini)")

    model_source = st.radio(
        "Gunakan model:",
        options=model_options,
        horizontal=True
    )

    if model_source == "Model Training Sendiri (sesi ini)":
        results = st.session_state.training_results
        st.success("✅ Menggunakan model dari sesi training.")
    else:
        saved = load_saved_model()
        if saved is None:
            st.error("⚠️ Model tidak ditemukan di `models/viral_classifier.pkl`. "
                    "Silakan train model terlebih dahulu lalu simpan.")
            return
        results = saved  # simpan dulu tanpa embedding
        st.success("✅ Model bawaan siap digunakan.")

    # ── Form prediksi (tidak berubah dari sebelumnya) ────────────
    st.markdown("""
    ### Masukkan Detail Tweet
    Sistem akan memprediksi probabilitas tweet Anda menjadi viral berdasarkan model yang sudah dilatih.
    """)
    st.markdown("---")

    with st.form("prediction_form"):
        st.subheader("Konten Tweet")
        tweet_text = st.text_area(
            "Masukkan teks tweet",
            height=150,
            max_chars=280,
            placeholder="Contoh: Jangan lewatkan promo spesial hari ini! Diskon hingga 70% untuk semua produk favorit kamu 🎉"
        )

        st.markdown("---")
        st.subheader("Informasi Akun")

        col1, col2, col3 = st.columns(3)
        with col1:
            followers = st.number_input("Jumlah Followers", min_value=0, max_value=1000000000, value=1000, step=100)
        with col2:
            following = st.number_input("Jumlah Following", min_value=0, max_value=1000000000, value=500, step=10)
        with col3:
            verified = st.selectbox(
                "Status Verifikasi",
                options=[("Tidak Terverifikasi", 0), ("Terverifikasi", 1)],
                format_func=lambda x: x[0]
            )

        st.markdown("---")
        submit_button = st.form_submit_button("Prediksi Viralitas", type="primary")

    if submit_button:
        if not tweet_text.strip():
            st.error("⚠️ Teks tweet tidak boleh kosong!")
        else:
            with st.spinner("Memproses prediksi..."):

                if 'model' not in results:
                    results = build_model_from_saved(results)
                    bert_tokenizer, bert_embedder, roberta_head = load_embedding_models()
                    results.update({
                        'bert_tokenizer': bert_tokenizer,
                        'bert_embedder':  bert_embedder,
                        'roberta_head':   roberta_head,
                    })

                tweet_len      = len(tweet_text)
                hashtags_count = tweet_text.count('#')
                mentions_count = tweet_text.count('@')

                result = predict_virality(
                    tweet_text, followers, following, verified[1],
                    tweet_len, hashtags_count, mentions_count,
                    results
                )

                prob_tidak_viral = result['probs'][0]
                prob_viral       = result['probs'][1]
                label            = result['label']

                st.markdown("---")
                st.success("✅ Prediksi Selesai!")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### Hasil Prediksi")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_viral * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilitas Viral (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30],  'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100],'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### Interpretasi")
                    if label == "viral":
                        st.success(f"""
                        **Tweet Diprediksi: VIRAL**

                        Probabilitas Viral: **{prob_viral*100:.2f}%**

                        Tweet Anda memiliki peluang besar untuk viral!
                        """)
                    else:
                        st.info(f"""
                        **Tweet Diprediksi: TIDAK VIRAL**

                        Probabilitas Viral: **{prob_viral*100:.2f}%**

                        Tweet ini kemungkinan tidak akan viral.
                        """)

                # Prediksi engagement
                st.markdown("---")
                st.markdown("### 📊 Prediksi Engagement")

                eng_col1, eng_col2, eng_col3, eng_col4 = st.columns(4)
                with eng_col1:
                    st.metric("Like",    f"~{result['like']:,}")
                with eng_col2:
                    st.metric("Retweet", f"~{result['retweet']:,}")
                with eng_col3:
                    st.metric("Quote",   f"~{result['quote']:,}")
                with eng_col4:
                    st.metric("Reply",   f"~{result['reply']:,}")

                st.markdown("---")
                st.markdown("### Detail Analisis")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Panjang Tweet", f"{tweet_len} char")
                with col2:
                    st.metric("Jumlah Hashtag", hashtags_count)
                with col3:
                    st.metric("Jumlah Mention", mentions_count)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Followers", f"{followers:,}")
                with col2:
                    st.metric("Following", f"{following:,}")
                with col3:
                    st.metric("Verified Status", verified[0])