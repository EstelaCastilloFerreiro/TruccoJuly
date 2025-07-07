#!/usr/bin/env python3
"""
Improved Model Training Script - Addresses overfitting and poor predictions
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.getcwd())

def main():
    print("🚀 Starting IMPROVED Sales Prediction Model")
    print("=" * 60)
    print("🔧 Focus: Preventing overfitting and improving predictions")
    print("=" * 60)
    
    try:
        # Step 1: Import required libraries
        print("📦 Importing libraries...")
        from catboost import CatBoostRegressor, Pool
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import joblib
        import json
        
        print("✅ Libraries imported successfully")
        
        # Step 2: Load your data
        print("\n📊 Loading data...")
        
        # Check if data file exists
        data_file = 'data/datos_modelo_catboost.xlsx'
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            print("Please ensure your data file is in the correct location")
            return
        
        df_ventas = pd.read_excel(data_file)
        print(f"✅ Data loaded: {len(df_ventas)} rows")
        
        # Step 3: Import the improved model functions
        print("\n🔧 Importing improved model functions...")
        
        from modelo import (
            prepare_data_with_validation,
            train_robust_model,
            predict_with_constraints
        )
        
        print("✅ Improved model functions imported")
        
        # Step 4: Prepare the data with validation
        print("\n🔄 Preparing data with validation...")
        df = df_ventas.copy()
        df['Fecha Documento'] = pd.to_datetime(df['Fecha Documento'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
        
        # Use improved data preparation
        en_temporada, fuera_temporada, monthly = prepare_data_with_validation(df, meses_futuros=None)
        
        print(f"✅ Data prepared with validation:")
        print(f"   - EN TEMPORADA: {len(en_temporada)} rows")
        print(f"   - FUERA TEMPORADA: {len(fuera_temporada)} rows")
        print(f"   - MONTHLY: {len(monthly)} rows")
        
        # Step 5: Define features and targets (simplified to prevent overfitting)
        print("\n🎯 Setting up features and targets...")
        
        target_en = 'Cantidad_en_temporada'
        target_fuera = 'Cantidad_fuera_temporada'
        
        # Simplified feature set to prevent overfitting
        base_features = [
            'mes', 'año', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'quarter', 'margen_unitario', 'precio_relativo', 'es_rebajas',
            'seasonality_index_verano', 'seasonality_index_invierno',
            'Tienda', 'Descripción Familia', 'Talla'
        ]
        
        # Add only the most important lag features
        lag_features_en = [
            'lag_1_cantidad_en_temporada', 'lag_3_cantidad_en_temporada', 
            'lag_12_cantidad_en_temporada',
            'rolling_6m_cantidad_en_temporada'
        ]
        
        lag_features_fuera = [
            'lag_1_cantidad_fuera_temporada', 'lag_3_cantidad_fuera_temporada', 
            'lag_12_cantidad_fuera_temporada',
            'rolling_6m_cantidad_fuera_temporada'
        ]
        
        features_en = base_features + lag_features_en
        features_fuera = base_features + ['Season'] + lag_features_fuera
        
        cat_features_en = ['Tienda', 'Descripción Familia', 'Talla', 'quarter']
        cat_features_fuera = ['Tienda', 'Descripción Familia', 'Talla', 'Season', 'quarter']
        
        print("✅ Simplified features defined to prevent overfitting")
        print(f"   - EN TEMPORADA features: {len(features_en)}")
        print(f"   - FUERA TEMPORADA features: {len(features_fuera)}")
        
        # Step 6: Prepare training data with proper time split
        print("\n📚 Preparing training data with time series split...")
        
        # Use only data up to June 2025 for training (more conservative)
        train_val_df_en = en_temporada[
            (en_temporada['año'] < 2025) | 
            ((en_temporada['año'] == 2025) & (en_temporada['mes'] <= 6))
        ].copy()
        
        train_val_df_fuera = fuera_temporada[
            (fuera_temporada['año'] < 2025) | 
            ((fuera_temporada['año'] == 2025) & (fuera_temporada['mes'] <= 6))
        ].copy()
        
        print(f"✅ Training data prepared:")
        print(f"   - EN TEMPORADA training: {len(train_val_df_en)} rows")
        print(f"   - FUERA TEMPORADA training: {len(train_val_df_fuera)} rows")
        
        # Step 7: Prepare features and targets
        y_en = train_val_df_en[target_en].copy()
        y_fuera = train_val_df_fuera[target_fuera].copy()
        
        X_en = train_val_df_en[features_en].copy()
        X_fuera = train_val_df_fuera[features_fuera].copy()
        
        # Fill missing values more conservatively
        X_en = X_en.fillna(0)
        X_fuera = X_fuera.fillna(0)
        
        # Remove rows with zero target (they don't help with prediction)
        non_zero_en = y_en > 0
        non_zero_fuera = y_fuera > 0
        
        X_en = X_en[non_zero_en]
        y_en = y_en[non_zero_en]
        X_fuera = X_fuera[non_zero_fuera]
        y_fuera = y_fuera[non_zero_fuera]
        
        print(f"✅ After filtering zero targets:")
        print(f"   - EN TEMPORADA: {len(X_en)} rows")
        print(f"   - FUERA TEMPORADA: {len(X_fuera)} rows")
        
        # Step 8: Train the robust models
        print("\n🤖 Training robust models with cross-validation...")
        
        # Create directories for saving models
        os.makedirs('modelos_mejorados', exist_ok=True)
        
        # Train EN TEMPORADA model
        print("\n" + "="*50)
        print("--- Training EN TEMPORADA Model ---")
        print("="*50)
        final_model_en = train_robust_model(X_en, y_en, cat_features_en, "EN TEMPORADA")
        
        # Train FUERA TEMPORADA model
        print("\n" + "="*50)
        print("--- Training FUERA TEMPORADA Model ---")
        print("="*50)
        final_model_fuera = train_robust_model(X_fuera, y_fuera, cat_features_fuera, "FUERA TEMPORADA")
        
        # Step 9: Save improved models
        print("\n💾 Saving improved models...")
        
        if final_model_en:
            joblib.dump(final_model_en['model'], "modelos_mejorados/model_en_robust.pkl")
            with open("modelos_mejorados/features_en_robust.json", "w") as f:
                json.dump({
                    "features": final_model_en['features'],
                    "cat_features": final_model_en['categorical_features'],
                    "model_name": final_model_en['model_name'],
                    "validation": final_model_en['validation'],
                    "prediction_constraints": final_model_en['prediction_constraints']
                }, f)
            print("✅ EN TEMPORADA robust model saved")
        
        if final_model_fuera:
            joblib.dump(final_model_fuera['model'], "modelos_mejorados/model_fuera_robust.pkl")
            with open("modelos_mejorados/features_fuera_robust.json", "w") as f:
                json.dump({
                    "features": final_model_fuera['features'],
                    "cat_features": final_model_fuera['categorical_features'],
                    "model_name": final_model_fuera['model_name'],
                    "validation": final_model_fuera['validation'],
                    "prediction_constraints": final_model_fuera['prediction_constraints']
                }, f)
            print("✅ FUERA TEMPORADA robust model saved")
        
        # Step 10: Final Summary with Overfitting Analysis
        print("\n" + "="*60)
        print("📊 IMPROVED MODEL SUMMARY")
        print("="*60)
        
        if final_model_en:
            print(f"\nEN TEMPORADA Model:")
            print(f"  Model: {final_model_en['model_name']}")
            print(f"  Features: {len(final_model_en['features'])}")
            print(f"  Final R²: {final_model_en['validation']['r2']:.4f}")
            print(f"  CV R²: {final_model_en['validation']['cv_r2']:.4f}")
            print(f"  RMSE: {final_model_en['validation']['rmse']:.4f}")
            print(f"  Overfitting Score: {final_model_en['validation']['overfitting_score']:.4f}")
            
            if final_model_en['validation']['overfitting_score'] > 0.1:
                print("  ⚠️ Potential overfitting detected")
            else:
                print("  ✅ No significant overfitting detected")
        
        if final_model_fuera:
            print(f"\nFUERA TEMPORADA Model:")
            print(f"  Model: {final_model_fuera['model_name']}")
            print(f"  Features: {len(final_model_fuera['features'])}")
            print(f"  Final R²: {final_model_fuera['validation']['r2']:.4f}")
            print(f"  CV R²: {final_model_fuera['validation']['cv_r2']:.4f}")
            print(f"  RMSE: {final_model_fuera['validation']['rmse']:.4f}")
            print(f"  Overfitting Score: {final_model_fuera['validation']['overfitting_score']:.4f}")
            
            if final_model_fuera['validation']['overfitting_score'] > 0.1:
                print("  ⚠️ Potential overfitting detected")
            else:
                print("  ✅ No significant overfitting detected")
        
        # Step 11: Test predictions on validation data
        print("\n🔮 Testing predictions on validation data...")
        
        if final_model_en and final_model_fuera:
            # Get recent data for validation
            recent_data = monthly[
                (monthly['año'] == 2025) & (monthly['mes'] > 6)
            ].head(10)
            
            if not recent_data.empty:
                print("Sample predictions for recent data:")
                
                # EN TEMPORADA predictions with constraints
                pred_en = predict_with_constraints(final_model_en, recent_data)
                
                # FUERA TEMPORADA predictions with constraints
                pred_fuera = predict_with_constraints(final_model_fuera, recent_data)
                
                for i in range(min(3, len(recent_data))):
                    real_en = recent_data.iloc[i].get('Cantidad_en_temporada', 0)
                    real_fuera = recent_data.iloc[i].get('Cantidad_fuera_temporada', 0)
                    
                    print(f"  Row {i+1}:")
                    print(f"    Real EN: {real_en:.2f}, Pred EN: {pred_en[i]:.2f}")
                    print(f"    Real FUERA: {real_fuera:.2f}, Pred FUERA: {pred_fuera[i]:.2f}")
        
        print("\n🎉 Improved model training completed successfully!")
        print("📁 Models saved in 'modelos_mejorados/' directory")
        print("\n🔧 Key improvements:")
        print("  - Time series cross-validation")
        print("  - Feature selection to prevent overfitting")
        print("  - Prediction constraints")
        print("  - Overfitting detection")
        print("  - Conservative model configurations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during improved model execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Improved model training completed successfully!")
    else:
        print("\n❌ Improved model training failed!") 