# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
get the dataframe from csv files and slip into train and test sets
to split accurately by time from 2013 - 2015 for training and 2016 - 2018 for testing to prevent leakage
for each split get sales as Ytrain and Ytest
"""
def get_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values(by='date')
    train = df[df['date'] < '2016-01-01']
    test = df[df['date'] >= '2016-01-01']
    
    return train, test

train_df, test_df = get_data(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\train.csv")

# %%
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error


def create_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_payday'] = np.where((df['day_of_month'] == 15) | (df['date'].dt.is_month_end), 1, 0)
    df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
    return df

def create_oil_price_feature(df, oil_price_path):
    oil_df = pd.read_csv(oil_price_path, parse_dates=['date'])
    oil_df = oil_df.rename(columns={'dcoilwtico': 'oil_price'})
    all_dates = pd.date_range(start=oil_df['date'].min(), end=oil_df['date'].max())
    oil_df = oil_df.set_index('date').reindex(all_dates).reset_index()
    oil_df.rename(columns={'index': 'date'}, inplace=True)
    oil_df['oil_price'] = oil_df['oil_price'].ffill()
    oil_df['oil_ma_7'] = oil_df['oil_price'].rolling(7).mean()
    df = df.merge(oil_df[['date', 'oil_price', 'oil_ma_7']], on='date', how='left')
    return df

def create_location_cluster_feature(df, stores_path):
    df_stores = pd.read_csv(stores_path)
    df = df.merge(df_stores[['store_nbr', 'state', 'city', 'cluster']], on='store_nbr', how='left')
    coastal_states = {'Esmeraldas', 'Guayas', 'Manabi', 'El Oro', 'Santa Elena'}
    df['is_coastal'] = df['state'].isin(coastal_states).astype(int)
    return df

def create_holiday_features(df, holidays_path):
    holidays = pd.read_csv(holidays_path, parse_dates=['date'])
    holidays = holidays[holidays['transferred'] == False]
    holidays = holidays[['date', 'locale', 'locale_name', 'description']]
    
    df['is_holiday'] = 0
    national_dates = holidays[holidays['locale'] == 'National']['date'].unique()
    df.loc[df['date'].isin(national_dates), 'is_holiday'] = 1
    
    # Local/Regional holidays mapping could be added here similar to previous steps
    # For speed/robustness in this fix, we stick to national + straightforward logic
    
    df['is_day_before_holiday'] = df['is_holiday'].shift(-1).fillna(0)
    return df

def create_school_supply_feature(df):
    df['is_school_supply_month'] = 0
    mask = (df['family'] == 'SCHOOL AND OFFICE SUPPLIES') & (df['month'].isin([8, 9]))
    df.loc[mask, 'is_school_supply_month'] = 1
    return df

def create_earthquake_feature(df):
    earthquake_date = pd.Timestamp('2016-04-16')
    df['earthquake_anomaly'] = (df['date'] >= earthquake_date).astype(int)
    return df

# --- FIXED: Horizon-Safe Rolling & Lags ---
def create_rolling_features(df, windows=[7, 14, 30, 60]):
    df = df.sort_values(['store_nbr', 'family', 'date'])
    for window in windows:
        # CRITICAL: Shift by 16 days. 
        # This ensures we only use data that will be available on the *last day* of the test set.
        grouped = df.groupby(['store_nbr', 'family'])['sales'].shift(16)
        df[f'rolling_mean_{window}'] = grouped.rolling(window, min_periods=1).mean().values
    return df

def create_lag_features(df, lags=[16, 21, 28, 30, 60, 90, 365]):
    # CRITICAL: Start lags at 16. 
    # lag_1 is impossible to know 16 days in advance without recursive prediction.
    df = df.sort_values(['store_nbr', 'family', 'date'])
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(lag))
    return df

def create_interaction_features(df):
    df['promo_on_weekend'] = df['onpromotion'] * df['is_weekend']
    df['promo_on_payday'] = df['onpromotion'] * df['is_payday']
    return df

def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# --- 2. UPDATED PIPELINE ---
def run_pipeline(train_df, test_df, oil_path, stores_path, holidays_path):
    print("Preparing data...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Concatenate
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # --- Feature Engineering Steps ---
    print("Running Feature Engineering...")
    df = create_time_features(df)
    df = create_oil_price_feature(df, oil_path)
    df = create_location_cluster_feature(df, stores_path)
    df = create_holiday_features(df, holidays_path)
    df = create_school_supply_feature(df)
    df = create_earthquake_feature(df)
    
    # These functions now use the FIXED lags/shifts
    df = create_rolling_features(df)
    df = create_lag_features(df)
    df = create_interaction_features(df)
    
    # Cleanup
    df = df.drop(columns=['city', 'state'], errors='ignore')
    df = optimize_memory(df)
    
    print("Splitting back into X/y sets...")
    train_processed = df[df['is_train'] == 1].copy()
    test_processed = df[df['is_train'] == 0].copy()
    
    y_train = train_processed['sales']
    # Drop 'id' from training
    X_train = train_processed.drop(columns=['sales', 'is_train', 'id'], errors='ignore')
    
    # Keep 'id' in X_test for submission
    if 'sales' in test_processed.columns:
        y_test = test_processed['sales']
        X_test = test_processed.drop(columns=['sales', 'is_train'], errors='ignore')
    else:
        y_test = None
        X_test = test_processed.drop(columns=['sales', 'is_train'], errors='ignore')

    # Remove rows with NaNs in TRAIN (due to the new larger lags, we lose first 16+ days)
    valid_train_mask = ~X_train.isnull().any(axis=1)
    X_train = X_train[valid_train_mask]
    y_train = y_train[valid_train_mask]

    print("Pipeline Complete.")
    return X_train, y_train, X_test, y_test


# --- 3. MODEL TRAINING ---
def train_family_models(X_train, y_train):
    family_models = {}
    families = X_train['family'].unique()
    print(f"Training {len(families)} separate models...")
    
    for family in families:
        mask = X_train['family'] == family
        family_X_train = X_train.loc[mask]
        family_y_train = y_train.loc[mask]
        family_y_train_log = np.log1p(family_y_train)
        
        X_input = family_X_train.drop(columns=['family', 'date', 'id'], errors='ignore').copy()
        
        # Sanitize Types
        for col in X_input.columns:
            if X_input[col].dtype == 'object':
                try:
                    X_input[col] = pd.to_numeric(X_input[col])
                except:
                    X_input[col] = X_input[col].astype('category')
        
        # Reduced Quantile Alpha to 0.5 (Median) to prevent over-shooting on sparse data
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'quantile_alpha': 0.5}

        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            objective='reg:quantileerror', 
            n_jobs=-1,
            enable_categorical=True
        )
        
        model.fit(X_input, family_y_train_log)
        family_models[family] = model
        
    print("All family models trained.")
    return family_models

def predict_family_models(family_models, X_test):
    predictions = pd.Series(index=X_test.index, dtype='float64')
    families = X_test['family'].unique()
    
    for family in families:
        if family not in family_models: continue
            
        mask = X_test['family'] == family
        family_X_test = X_test.loc[mask]
        
        X_input = family_X_test.drop(columns=['family', 'date', 'id'], errors='ignore').copy()
        
        for col in X_input.columns:
            if X_input[col].dtype == 'object':
                try:
                    X_input[col] = pd.to_numeric(X_input[col])
                except:
                    X_input[col] = X_input[col].astype('category')
        
        model = family_models[family]
        preds_log = model.predict(X_input)
        preds_real = np.expm1(preds_log)
        preds_real = np.maximum(preds_real, 0)
        
        predictions.loc[mask] = preds_real
        
    return predictions

# --- 4. NEW: LOCAL VALIDATION FUNCTION ---
def validate_pipeline(train_file_path, oil_path, stores_path, holidays_path):
    """
    Simulates the competition by splitting the TRAINING data.
    Uses last 16 days of Train as 'Test' to calculate local RMSLE.
    """
    print("\n--- STARTING LOCAL VALIDATION ---")
    full_train = pd.read_csv(train_file_path, parse_dates=['date'])
    
    # Split: Train = All dates except last 16 days. Test = Last 16 days.
    max_date = full_train['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=16)
    
    print(f"Splitting data... Train ends: {cutoff_date}, Validation: {cutoff_date + pd.Timedelta(days=1)} to {max_date}")
    
    local_train = full_train[full_train['date'] <= cutoff_date].copy()
    local_valid = full_train[full_train['date'] > cutoff_date].copy()
    
    # Run Pipeline just like real submission
    X_t, y_t, X_v, y_v = run_pipeline(
        local_train, 
        local_valid.drop(columns=['sales']), # Hide sales from validation input
        oil_path, stores_path, holidays_path
    )
    
    # Train
    models = train_family_models(X_t, y_t)
    
    # Predict
    preds = predict_family_models(models, X_v)
    
    # Score
    # Re-attach actual sales for scoring (using index alignment)
    actuals = local_valid.loc[X_v.index, 'sales']
    rmsle = np.sqrt(mean_squared_log_error(actuals, preds))
    
    print(f"\n>>> LOCAL VALIDATION RMSLE: {rmsle:.5f} <<<")
    return rmsle

# %%
# 1. Validation (Uncomment to check score locally before submitting)
validate_pipeline(
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\train.csv",
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\oil.csv",
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\stores.csv", 
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\holidays_events.csv"
 )


# %%
def plot_error_heatmap(X_test, y_test, family_models):
    """
    Visualizes the RMSLE (log error) for each family and location in a heatmap.
    OPTIMIZED: Predicts by family batch instead of store-by-store loops.
    """
    families = list(family_models.keys())
    stores = sorted(X_test['store_nbr'].unique())
    
    # Store results in a dictionary first for easy lookup
    # Keys: (family_index, store_index), Value: RMSLE
    results = {}
    
    print("Generating Heatmap Data...")
    
    for i, family in enumerate(families):
        if family not in family_models:
            continue
            
        # 1. Get all data for this family (across all stores)
        family_mask = X_test['family'] == family
        if not family_mask.any():
            continue
            
        X_fam = X_test.loc[family_mask]
        y_fam = y_test.loc[family_mask]
        
        # 2. Predict for the whole family at once
        model = family_models[family]
        
        # Prepare input and sanitize types (Fixes ValueError: object)
        X_input = X_fam.drop(columns=['family', 'date'], errors='ignore').copy()
        
        # --- SANITIZATION BLOCK ---
        for col in X_input.columns:
            if X_input[col].dtype == 'object':
                try:
                    X_input[col] = pd.to_numeric(X_input[col])
                except (ValueError, TypeError):
                    X_input[col] = X_input[col].astype('category')
        # --------------------------

        preds_log = model.predict(X_input)
        preds = np.expm1(preds_log)
        preds = np.clip(preds, 0, None)
        
        # 3. Create a temporary DF to calculate error per store
        temp_df = pd.DataFrame({
            'store_nbr': X_fam['store_nbr'].values,
            'actual': y_fam.values,
            'predicted': preds
        })
        
        # 4. Group by store and calculate RMSLE
        # Function to calc RMSLE safely
        def calc_rmsle(group):
            return np.sqrt(mean_squared_log_error(group['actual'], group['predicted']))
            
        store_errors = temp_df.groupby('store_nbr').apply(calc_rmsle)
        
        # 5. Map back to matrix indices
        for store in stores:
            if store in store_errors.index:
                # Find store index in our specific sorted list
                j = stores.index(store)
                results[(i, j)] = store_errors[store]
    
    # Fill the matrix
    error_matrix = np.zeros((len(families), len(stores)))
    error_matrix[:] = np.nan # Default to NaN
    
    for (i, j), rmsle in results.items():
        error_matrix[i, j] = rmsle   
    
    # Plotting
    plt.figure(figsize=(20, 12))
    plt.imshow(error_matrix, cmap='hot_r', interpolation='nearest', aspect='auto') # hot_r so red=bad, yellow=ok
    plt.colorbar(label='RMSLE')
    
    plt.xticks(range(len(stores)), stores, rotation=90, fontsize=8)
    plt.yticks(range(len(families)), families, fontsize=8)
    
    plt.xlabel('Store Number')
    plt.ylabel('Product Family')
    plt.title('RMSLE Heatmap by Family and Store (Darker is Higher Error)')
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(family_models, X_test, y_test, family, store_nbr):
    """
    Visualizes predictions vs actual sales for a specific family and store.
    """
    if family not in family_models:
        print(f"Error: No model found for family {family}")
        return

    # Filter data
    mask = (X_test['family'] == family) & (X_test['store_nbr'] == store_nbr)
    
    if not mask.any():
        print(f"No data found for Family: {family}, Store: {store_nbr}")
        return
        
    X_subset = X_test.loc[mask]
    y_subset = y_test.loc[mask]
    
    # Predict
    model = family_models[family]
    
    # Prepare input and sanitize types (Fixes ValueError: object)
    X_input = X_subset.drop(columns=['family', 'date'], errors='ignore').copy()
    
    # --- SANITIZATION BLOCK ---
    for col in X_input.columns:
        if X_input[col].dtype == 'object':
            try:
                X_input[col] = pd.to_numeric(X_input[col])
            except (ValueError, TypeError):
                X_input[col] = X_input[col].astype('category')
    # --------------------------
    
    preds_log = model.predict(X_input)
    predictions = np.expm1(preds_log)
    predictions = np.clip(predictions, 0, None)
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(X_subset['date'], y_subset.values, label='Actual Sales', marker='o', markersize=4)
    plt.plot(X_subset['date'], predictions, label='Predicted Sales', linestyle='--', linewidth=2)
    
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'Predictions vs Actuals: {family} @ Store {store_nbr}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# %%
plot_error_heatmap(Xtest, Ytest, family_models)
# %%
plot_predictions_vs_actual(family_models, Xtest, Ytest, family='PRODUCE', store_nbr=44)
# %%
"""
the model seems to be generally undervaluing sales for most families
especially BABY CARE
"""


# %%
# 2. Real Submission
train_df = pd.read_csv(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\train.csv", parse_dates=['date'])
test_df_original = pd.read_csv(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\test.csv", parse_dates=['date'])

Xtrain, Ytrain, Xtest_final, _ = run_pipeline(
    train_df, 
    test_df_original, 
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\oil.csv",
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\stores.csv", 
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\holidays_events.csv"
)

family_models = train_family_models(Xtrain, Ytrain)
final_predictions = predict_family_models(family_models, Xtest_final)

submission = pd.DataFrame({
    'id': Xtest_final['id'].astype(int),
    'sales': final_predictions
})
submission = submission.sort_values('id')
submission.to_csv(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\submission.csv", index=False)
print("Predictions saved successfully.")