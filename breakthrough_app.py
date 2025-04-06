import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

# Set page config
st.set_page_config(page_title="Breakthrough Time Predictor", layout="wide")

# Title and description
st.title("Breakthrough Time Prediction for Adsorption Columns")
st.markdown("""
This app predicts breakthrough times using machine learning models trained on adsorption column data.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

def user_input_features():
    q = st.sidebar.number_input("Flow Rate (Q in mL/min)", min_value=0.1, value=4.0, step=0.1)
    z = st.sidebar.number_input("Bed Height (Z in cm)", min_value=1.0, value=10.0, step=0.1)
    c0 = st.sidebar.number_input("Initial Concentration (C₀ in mg/L)", min_value=1.0, value=250.0, step=1.0)
    mtz = st.sidebar.number_input("Mass Transfer Zone (cm)", min_value=0.1, value=9.5, step=0.1)
    mass = st.sidebar.number_input("Mass of Adsorbent (g)", min_value=0.1, value=10.71, step=0.01)
    
    # Calculate engineered features
    q_z = q / z
    mtz_z = mtz / z
    q_mtz = q * mtz
    adsorption_capacity = mass / (q * z)
    
    data = {
        'Q (mL/min)': q,
        'Z (cm)': z,
        'C₀ (mg/L)': c0,
        'Mass Transfer Zone (cm)': mtz,
        'Mass of Adsorbent (g)': mass,
        'Q/Z': q_z,
        'MTZ/Z': mtz_z,
        'Q*MTZ': q_mtz,
        'Adsorption_Capacity': adsorption_capacity
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Load and preprocess data with error handling
@st.cache_data
def load_data():
    try:
        file_path = "data/Dataset-Breakthroughtime.xlsx"
        
        # Verify file exists
        if not os.path.exists(file_path):
            st.error(f"Critical Error: File not found at {file_path}")
            st.error("Please ensure:")
            st.error("1. A 'data' folder exists")
            st.error("2. It contains 'Dataset-Breakthroughtime.xlsx'")
            st.error("3. Filename is EXACTLY as shown (case-sensitive)")
            st.stop()
            
        df = pd.read_excel(file_path)
        
        # Preprocessing
        df = df.dropna()
        df["Q/Z"] = df["Q (mL/min)"] / df["Z (cm)"]
        df["MTZ/Z"] = df["Mass Transfer Zone (cm)"] / df["Z (cm)"]
        df["Q*MTZ"] = df["Q (mL/min)"] * df["Mass Transfer Zone (cm)"]
        df["Adsorption_Capacity"] = df["Mass of Adsorbent (g)"] / (df["Q (mL/min)"] * df["Z (cm)"])
        
        df = df[(df["Breakthrough Time (min)"] > 0) & 
                (df["Mass Transfer Zone (cm)"] <= df["Z (cm)"])]
        
        return df
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# Train models
@st.cache_resource
def train_models(df):
    features = ['Q (mL/min)', 'Z (cm)', 'Q/Z', 'MTZ/Z', 'Q*MTZ', 'Adsorption_Capacity']
    X = df[features]
    y = df["Breakthrough Time (min)"]
    
    # Time-series aware split
    test_size = int(len(df) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_features=0.5,
        bootstrap=False,
        random_state=42
    )
    
    # XGBoost
    xgbr = xgb.XGBRegressor(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    
    # Stacked Ensemble
    stacked = StackingRegressor(
        estimators=[
            ('gb', gb),
            ('rf', rf),
            ('xgb', xgbr)
        ],
        final_estimator=MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        ),
        cv=5
    )
    
    models = {
        "Gradient Boosting": gb,
        "Random Forest": rf,
        "XGBoost": xgbr,
        "Stacked Ensemble": stacked
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models, X_test, y_test

# Main app
def main():
    df = load_data()
    models, X_test, y_test = train_models(df)
    
    # Get user input
    input_df = user_input_features()
    
    # Display user inputs
    st.subheader("User Input Parameters")
    st.dataframe(input_df)
    
    # Make predictions
    st.subheader("Model Predictions")
    
    features = ['Q (mL/min)', 'Z (cm)', 'Q/Z', 'MTZ/Z', 'Q*MTZ', 'Adsorption_Capacity']
    input_features = input_df[features]
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_features)[0]
    
    # Display predictions
    pred_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Breakthrough Time (min)'])
    st.dataframe(pred_df.style.format("{:.2f}"))
    
    # Determine best model (highest R²)
    model_performance = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        model_performance[name] = r2_score(y_test, y_pred)
    
    best_model = max(model_performance, key=model_performance.get)
    best_prediction = predictions[best_model]
    
    st.subheader(f"Recommended Prediction (from {best_model})")
    st.metric(label="Predicted Breakthrough Time", value=f"{best_prediction:.2f} minutes")
    
    # Plot all predictions
    st.subheader("Prediction Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(predictions.keys(), predictions.values())
    ax.set_ylabel("Breakthrough Time (min)")
    ax.set_title("Model Predictions Comparison")
    st.pyplot(fig)
    
    # Plot actual vs predicted for test data
    st.subheader("Model Performance on Test Data")
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        axes[i].scatter(y_test, y_pred, alpha=0.5)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(f"{name} (R²={model_performance[name]:.2f})")
    
    plt.tight_layout()
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
