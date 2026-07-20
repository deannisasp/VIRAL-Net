# Models Directory

Letakkan trained models di sini:

## Classification Models
- `viralnet_classifier.pth` - PyTorch classification model
- `scaler.pkl` - MinMax scaler untuk metadata
- `label_encoder.pkl` - Label encoder untuk classes

## Forecasting Models
- `arima_model.pkl` - ARIMA model
- `sarima_model.pkl` - Seasonal ARIMA model
- `prophet_model.pkl` - Facebook Prophet model
- `lstm_model.h5` - LSTM neural network
- `xgboost_model.pkl` - XGBoost model
- `ensemble_weights.json` - Ensemble configuration

## Tokenizer
- `indobert_tokenizer/` - IndoBERT tokenizer files (auto-downloaded jika belum ada)

## Notes
- Model files tidak di-commit ke git (terlalu besar)
- Download dari Google Drive atau train ulang dari notebook
- Pastikan path model sesuai dengan konfigurasi di `utils/model_loader.py`
