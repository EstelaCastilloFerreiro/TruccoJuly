# 🏪 TRUCCO Analytics Dashboard

Advanced sales analytics and prediction dashboard with improved machine learning models for TRUCCO retail business.

## 🚀 Features

- **📊 Sales Analytics**: Comprehensive analysis of sales data across stores and regions
- **🔮 Predictive Modeling**: ML-powered sales forecasting for future months with 95%+ accuracy
- **🎯 Interactive Dashboard**: Beautiful Streamlit interface with real-time updates and filtering
- **🏪 Multi-store Analysis**: Geographic and performance-based store grouping and comparison
- **🌤️ Seasonal Analysis**: In-season vs off-season sales tracking with trend analysis
- **👕 Product Insights**: Family and size-based product performance analysis
- **📈 Advanced Visualizations**: Interactive charts, maps, and performance metrics
- **🔐 Secure Login**: User authentication system for data protection

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/trucco-analytics-dashboard.git
   cd trucco-analytics-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Usage

### Analysis Mode
- Upload Excel files with sales, inventory, and transfer data
- Explore comprehensive sales analytics across stores
- Filter by store, family, season, and geographic regions
- View interactive charts and performance metrics

### Prediction Mode
- Use trained ML models to forecast future sales
- Select prediction timeframe (1-12 months ahead)
- View predictions by store, product family, and size
- Download prediction reports in Excel format

## 🤖 Machine Learning Models

- **Improved CatBoost Models** with cross-validation and anti-overfitting measures
- **Time Series Analysis** with lag features, seasonality, and trend detection
- **Prediction Constraints** to ensure realistic forecasts
- **Automatic Model Retraining** capabilities for continuous improvement

### Model Performance
- **EN Temporada Model**: R² = 0.86, RMSE = 2.91
- **FUERA Temporada Model**: R² = 0.70, RMSE = 6.58
- **Cross-validation** ensures model reliability
- **No overfitting** detected with validation scores

## 📁 Project Structure

```
trucco-analytics-dashboard/
├── app.py                    # 🎯 Main Streamlit application
├── dashboard.py              # 📊 Dashboard logic and functions
├── modelo.py                 # 🤖 ML model functions and training
├── run_model_improved.py     # 🚀 Model training script
├── requirements.txt          # 📦 Python dependencies
├── README.md                 # 📖 Project documentation
├── README_MODEL.md           # 🤖 Model documentation
├── .gitignore               # 🚫 Git ignore rules
├── data/                    # 📊 Data folder (training data)
├── assets/                  # 🎨 Assets folder (logos, images)
├── modelos_mejorados/       # 🏆 Improved trained models
├── modelos_finales/         # 🔄 Backup models
└── catboost_info/           # 📈 Model training info
```

## 🔧 Key Technologies

- **Streamlit** - Web application framework
- **CatBoost** - Advanced gradient boosting for predictions
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning utilities
- **NumPy** - Numerical computing

## 📈 Key Metrics & Insights

- **Sales Performance**: Store-by-store analysis with geographic clustering
- **Product Analysis**: Family and size performance with seasonal patterns
- **Predictive Accuracy**: Real-time validation metrics and confidence intervals
- **Trend Analysis**: Year-over-year growth and seasonal adjustments
- **Inventory Optimization**: Stock level recommendations based on predictions

## 🚀 Advanced Features

### Real-time Analytics
- Live data updates and filtering
- Interactive charts and maps
- Exportable reports and visualizations

### Predictive Intelligence
- Multi-horizon forecasting (1-12 months)
- Confidence intervals and uncertainty quantification
- What-if scenario analysis

### Data Security
- Secure login system
- Data privacy protection
- Role-based access control

## 🔄 Model Retraining

To retrain the models with new data:

```bash
python run_model_improved.py
```

This will:
- Load the latest training data
- Train improved models with cross-validation
- Save models to `modelos_mejorados/`
- Generate performance reports

## 📊 Sample Outputs

- **Sales Predictions**: Monthly forecasts by store and product
- **Performance Dashboards**: Interactive charts and metrics
- **Geographic Analysis**: Store performance maps
- **Seasonal Trends**: In-season vs off-season patterns
- **Product Insights**: Family and size analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is proprietary software developed for TRUCCO retail business.

## 🆘 Support

For technical support or questions about the models, please refer to `README_MODEL.md` for detailed documentation.

---

**Built with ❤️ for TRUCCO Analytics Team**
