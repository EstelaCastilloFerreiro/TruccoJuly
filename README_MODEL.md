# Sales Prediction Model - Usage Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Ensure your data file `datos_modelo_catboost.xlsx` is in one of these locations:
- Current directory: `./datos_modelo_catboost.xlsx`
- Data folder: `./data/datos_modelo_catboost.xlsx`
- Google Drive (if using Colab): `/content/drive/MyDrive/TRUCCO/COFITECH/data/datos_modelo_catboost.xlsx`

### 3. Run the Model
```bash
python run_model_improved.py
```

## 📁 What the Model Does

The improved model includes:

### 🔧 **Enhanced Features**
- **Time Series Features**: Multiple lags (1, 2, 3, 6, 12, 24 months)
- **Rolling Windows**: 3, 6, 12-month averages and trends
- **Seasonality**: Sine/cosine transformations, seasonal indices
- **Trend Analysis**: Short vs long-term trends
- **Volatility**: Rolling standard deviations
- **Year-over-Year**: Growth rates and deltas

### 🤖 **Advanced Model Training**
- **CatBoost Regressor** with optimized hyperparameters
- **Early Stopping** to prevent overfitting
- **Cross-Validation** for robust evaluation
- **Feature Selection** to remove noise
- **Ensemble Methods** for better predictions

### 📊 **Two Separate Models**
1. **EN TEMPORADA Model**: Predicts in-season sales
2. **FUERA TEMPORADA Model**: Predicts off-season sales

## 📈 Model Outputs

### Saved Files
- `modelos_mejorados/model_en_robust.pkl` - EN TEMPORADA improved model
- `modelos_mejorados/model_fuera_robust.pkl` - FUERA TEMPORADA improved model
- `modelos_mejorados/features_en_robust.json` - Feature configuration
- `modelos_mejorados/features_fuera_robust.json` - Feature configuration

**Note**: The improved models include cross-validation, feature selection, and anti-overfitting measures for better performance.

### Performance Metrics
- **R² Score**: How well the model explains variance
- **RMSE**: Root Mean Square Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## 🔍 Troubleshooting

### Common Issues

1. **Data File Not Found**
   ```
   ❌ Data file not found. Please ensure 'datos_modelo_catboost.xlsx' is in the correct location
   ```
   **Solution**: Move your data file to the current directory or update the path in the script.

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'catboost'
   ```
   **Solution**: Install dependencies: `pip install -r requirements.txt`

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce data size or use a machine with more RAM.

4. **Feature Mismatch**
   ```
   KeyError: 'feature_name' not found
   ```
   **Solution**: Check that your data contains all required columns.

### Data Requirements

Your Excel file should contain these columns:
- `Fecha Documento` (date format: DD/MM/YYYY)
- `Tienda` (store name)
- `Descripción Familia` (product family)
- `Talla` (size)
- `Cantidad` (quantity sold)
- `Importe` (amount)
- `Margen` (margin)
- `Season` (season information)

## 🎯 Using the Trained Models

### Load and Use Models
```python
import joblib
import json

# Load improved models (recommended)
model_en = joblib.load('modelos_mejorados/model_en_robust.pkl')
model_fuera = joblib.load('modelos_mejorados/model_fuera_robust.pkl')

# Load feature configurations
with open('modelos_mejorados/features_en_robust.json', 'r') as f:
    config_en = json.load(f)

with open('modelos_mejorados/features_fuera_robust.json', 'r') as f:
    config_fuera = json.load(f)

# Make predictions with constraints
# (Use the improved prediction functions from dashboard.py)
```

## 📞 Support

If you encounter issues:
1. Check the error messages carefully
2. Ensure all dependencies are installed
3. Verify your data format matches requirements
4. Check available memory and disk space

## 🎉 Success Indicators

When the model runs successfully, you should see:
- ✅ All libraries imported
- ✅ Data loaded with row count
- ✅ Data prepared with EN/FUERA/MONTHLY counts
- ✅ Models trained with performance metrics
- ✅ Models saved to `modelos_finales/` directory
- 🎯 "Model is ready for use!" message 