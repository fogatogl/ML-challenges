# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
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

def optimize_memory(df):
    """
    Optional: Reduces memory usage by downcasting data types.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def create_time_features(df):
    """
    VECTORIZED: approx 50x faster than .apply()
    """
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    # Vectorized check for payday
    # (15th, or last day of month which is usually 30 or 31)
    df['is_payday'] = np.where(
        (df['day_of_month'] == 15) | (df['date'].dt.is_month_end), 
        1, 0
    )
    
    # NEW: Weekend feature (Friday=4, Saturday=5, Sunday=6) - Adjust based on local culture
    # In Ecuador, shopping spikes on weekends.
    df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
    
    return df

def create_lag_features(df, lags=[7, 14, 28]):
    """
    CORRECTED: Now groups by store and family before shifting.
    Prevents data leakage between different products.
    """
    # Ensure data is sorted for correct shifting
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    for lag in lags:
        # transform() keeps the index aligned
        df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(lag))
        
    return df

def create_rolling_features(df, windows=[7, 30]):
    """
    NEW FEATURE: Rolling Averages.
    Helps the model see the 'Trend' rather than just noisy daily spikes.
    """
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    for window in windows:
        # We shift by 1 first to avoid leakage (cannot use today's sales to predict today)
        # min_periods=1 ensures we get values even at the start of the window
        grouped = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
        df[f'rolling_mean_{window}'] = grouped.rolling(window, min_periods=1).mean().values
        
    return df

def create_oil_price_feature(df, oil_price_path):
    """
    IMPROVED: Handles weekend gaps using forward fill.
    """
    oil_df = pd.read_csv(oil_price_path, parse_dates=['date'])
    oil_df = oil_df.rename(columns={'dcoilwtico': 'oil_price'}) # Normalized name
    
    # 1. Reindex to handle missing weekends in oil data
    all_dates = pd.date_range(start=oil_df['date'].min(), end=oil_df['date'].max())
    oil_df = oil_df.set_index('date').reindex(all_dates).reset_index()
    oil_df.rename(columns={'index': 'date'}, inplace=True)
    
    # 2. Forward Fill: Friday's price applies to Sat/Sun
    oil_df['oil_price'] = oil_df['oil_price'].ffill()
    
    # 3. Create a Moving Average for oil (External economic factors lag behind)
    oil_df['oil_ma_7'] = oil_df['oil_price'].rolling(7).mean()
    
    df = df.merge(oil_df[['date', 'oil_price', 'oil_ma_7']], on='date', how='left')
    return df

def create_school_supply_feature(df):
    """
    VECTORIZED: Uses boolean indexing instead of row-by-row lambda.
    """
    # Initialize column with 0
    df['is_school_supply_month'] = 0
    
    # Condition: Family is School/Office AND Month is Aug/Sept
    mask = (df['family'] == 'SCHOOL AND OFFICE SUPPLIES') & (df['month'].isin([8, 9])) # Coastal region school start
    
    # Vectorized assignment
    df.loc[mask, 'is_school_supply_month'] = 1
    
    return df

def create_earthquake_feature(df):
    """
    CORRECTED: The Ecuador earthquake was April 16, 2016.
    """
    mask = (df['date'] >= '2016-04-16') & (df['date'] <= '2016-05-16')
    df = df[~mask] 
    return df

def create_location_cluster_feature(df, stores_path):
    """
    VECTORIZED: Uses isin() for instant lookup.
    """
    df_stores = pd.read_csv(stores_path)
    
    # Define lists
    coastal_states = {'Esmeraldas', 'Guayas', 'Manabi', 'El Oro', 'Santa Elena'}
    urban_states = {'Pichincha', 'Guayas', 'Azuay', 'Loja', 'Tungurahua'}
    
    # Merge store info first
    # Using 'left' merge preserves the main df structure
    df = df.merge(df_stores[['store_nbr', 'state', 'city', 'cluster']], on='store_nbr', how='left')
    
    # Vectorized creation
    df['is_coastal'] = df['state'].isin(coastal_states).astype(int)
    df['is_urban'] = df['state'].isin(urban_states).astype(int)
    
    # CRITICAL FIX: Do NOT drop 'state' and 'city' here.
    # The 'create_holiday_features' function needs them later to map local holidays.
    # We will drop them in 'run_pipeline' at the very end.    
    return df

def create_holiday_features(df, holidays_path):
    """
    Maps National, Regional, and Local holidays to the correct stores.
    """
    holidays = pd.read_csv(holidays_path, parse_dates=['date'])
    
    # 1. Filter out transferred holidays (they aren't holidays anymore)
    holidays = holidays[holidays['transferred'] == False]
    
    # 2. Treat 'Bridge' and 'Transfer' days as real holidays
    # We only care about the date and the locale
    holidays = holidays[['date', 'locale', 'locale_name', 'description']]
    
    # 3. Create a unified holiday boolean column
    # Initialize as 0
    df['is_holiday'] = 0
    df['holiday_type'] = 0 # 0=None, 1=National, 2=Regional/Local
    
    # A. Map National Holidays (Apply to everyone)
    national_dates = holidays[holidays['locale'] == 'National']['date'].unique()
    df.loc[df['date'].isin(national_dates), 'is_holiday'] = 1
    df.loc[df['date'].isin(national_dates), 'holiday_type'] = 1
    
    # B. Map Regional/Local Holidays (Complex join)
    # This requires 'state' and 'city' to still be in df (from location_cluster function)
    
    # Local (City match)
    # Explicitly rename description to prevent merge confusion if column doesn't exist
    local_holidays = holidays[holidays['locale'] == 'Local'][['date', 'locale_name', 'description']]
    local_holidays = local_holidays.rename(columns={'description': 'description_local'})
    
    # Merge on date AND city name
    df = df.merge(local_holidays, 
                  left_on=['date', 'city'], 
                  right_on=['date', 'locale_name'], 
                  how='left')
    
    # Regional (State match)
    regional_holidays = holidays[holidays['locale'] == 'Regional'][['date', 'locale_name', 'description']]
    regional_holidays = regional_holidays.rename(columns={'description': 'description_regional'})
    
    df = df.merge(regional_holidays, 
                  left_on=['date', 'state'], 
                  right_on=['date', 'locale_name'], 
                  how='left', suffixes=('', '_reg'))
    
    # Update is_holiday flag if we found a match
    # We check if columns exist just to be safe, but they should be there now
    if 'description_local' in df.columns:
        df['is_holiday'] = df['is_holiday'] | df['description_local'].notna()
        
    if 'description_regional' in df.columns:
        df['is_holiday'] = df['is_holiday'] | df['description_regional'].notna()
    
    # Clean up merge artifacts
    cols_to_drop = ['locale_name', 'description_local', 'locale_name_reg', 'description_regional']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 4. CRITICAL: "The Day Before" Feature
    # Liquor sales spike the day BEFORE a holiday.
    df['is_day_before_holiday'] = df['is_holiday'].shift(-1).fillna(0)
    
    # 5. "Days to Next Holiday" (Advanced)
    # Helps the model anticipate the spike
    # (Simplified version: binary flag for "Holiday is coming in 3 days")
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
    df['holiday_coming_soon'] = df['is_holiday'].rolling(window=indexer, min_periods=1).max()
    
    return df

def create_produce_specific_features(df):
    """
    Targets the high-variance nature of the PRODUCE family.
    """
    # 1. Family-Specific Rolling Promotions
    # Is the store currently in a 'heavy' promo period for produce?
    df['produce_promo_rolling_7'] = df.groupby(['store_nbr', 'family'])['onpromotion'].transform(lambda x: x.rolling(7).mean())
    
    # 2. Anticipation: Is there a big promo TOMORROW?
    # People skip buying produce today if they know it's on sale tomorrow.
    df['onpromotion_tomorrow'] = df.groupby(['store_nbr', 'family'])['onpromotion'].shift(-1).fillna(0)

    # 3. Market Day Indicator (The "Tuesday" Effect)
    # Many Ecuadorian stores have 'Fresh Tuesdays'. 
    # Let's see if Tuesdays generally have higher sales for Produce.
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['produce_tuesday_promo'] = df['is_tuesday'] * df['onpromotion']

    # 4. Sales Momentum (Ratio of short-term vs long-term)
    # If the 3-day average is much higher than the 30-day average, we are in a peak.
    # Note: Using lags to avoid leakage
    short_roll = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(1).rolling(3).mean())
    long_roll = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(1).rolling(14).mean())
    df['sales_momentum'] = short_roll / (long_roll + 1)
    
    return df


def create_interaction_features(df):
    """
    CRITICAL FOR: PRODUCE and HOME CARE
    Captures "Weekend Markets" and "Payday Shopping Sprees".
    """
    # 1. Weekend + Promo (The "Market Day" effect for Produce)
    # Produce sells 5x more if it is a Weekend AND on Promotion
    df['promo_on_weekend'] = df['onpromotion'] * df['is_weekend']
    
    # 2. Payday + Promo (The "Stock Up" effect for Home Care/Liquor)
    # People buy expensive items (Liquor) or bulk items (Home Care) when they have cash
    df['promo_on_payday'] = df['onpromotion'] * df['is_payday']
    
    return df

# 1. UPDATED PIPELINE: KEEPS 'id' IN X_TEST
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
    
    # Note: These steps sort the dataframe! This is why ID tracking is crucial.
    df = create_rolling_features(df)
    df = create_lag_features(df)
    df = create_interaction_features(df)
    
    # Add Produce Features if you included that function
    if 'create_produce_specific_features' in globals():
        df = create_produce_specific_features(df)
    
    # Cleanup
    df = df.drop(columns=['city', 'state'], errors='ignore')
    df = optimize_memory(df)
    
    print("Splitting back into X/y sets...")
    train_processed = df[df['is_train'] == 1].copy()
    test_processed = df[df['is_train'] == 0].copy()
    
    y_train = train_processed['sales']
    # Drop 'id' from training because we don't need it there
    X_train = train_processed.drop(columns=['sales', 'is_train', 'id'], errors='ignore')
    
    # Keep 'id' in X_test
    if 'sales' in test_processed.columns:
        y_test = test_processed['sales']
        X_test = test_processed.drop(columns=['sales', 'is_train'], errors='ignore')
    else:
        y_test = None
        X_test = test_processed.drop(columns=['sales', 'is_train'], errors='ignore')

    # Remove rows with NaNs in TRAIN (due to lags)
    valid_train_mask = ~X_train.isnull().any(axis=1)
    X_train = X_train[valid_train_mask]
    y_train = y_train[valid_train_mask]

    print("Pipeline Complete.")
    return X_train, y_train, X_test, y_test

Xtrain, Ytrain, Xtest, Ytest = run_pipeline(
     train_df,test_df, r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\oil.csv",
     r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\stores.csv", r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\holidays_events.csv")
# %%
def improve_produce_model_params(family):
    """
    Logic for tuning PRODUCE specifically.
    PRODUCE usually benefits from a higher 'learning_rate' and 'max_depth' 
    compared to stable categories like 'STATIONERY'.
    """
    if family == 'PRODUCE':
        return {
            'n_estimators': 300,        # More trees for complex patterns
            'learning_rate': 0.05,      # Slower learning to capture noise
            'max_depth': 8,             # Deeper trees for interaction effects
            'colsample_bytree': 0.9,
            'quantile_alpha': 0.5       # PRODUCE is less biased than Liquor, use 0.5 (Median)
        }
    else:
        # Default params for others
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'quantile_alpha': 0.6
        }

def train_family_models(X_train, y_train):
    family_models = {}
    families = X_train['family'].unique()
    print(f"Training {len(families)} separate models...")
    
    for family in families:
        mask = X_train['family'] == family
        family_X_train = X_train.loc[mask]
        family_y_train = y_train.loc[mask]
        family_y_train_log = np.log1p(family_y_train)
        
        # Explicitly drop 'date', 'family', and 'id' if present
        X_input = family_X_train.drop(columns=['family', 'date', 'id'], errors='ignore').copy()
        
        # Sanitize Types
        for col in X_input.columns:
            if X_input[col].dtype == 'object':
                try:
                    X_input[col] = pd.to_numeric(X_input[col])
                except:
                    X_input[col] = X_input[col].astype('category')
        
        # Check for custom params (Produce optimization)
        if 'improve_produce_model_params' in globals():
            params = improve_produce_model_params(family)
        else:
            params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'quantile_alpha': 0.6}

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
        
        # Explicitly drop 'date', 'family', AND 'id' just for the model
        X_input = family_X_test.drop(columns=['family', 'date', 'id'], errors='ignore').copy()
        
        # Sanitize Types
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


family_models = train_family_models(Xtrain, Ytrain)
predictions = predict_family_models(family_models, Xtest)

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
# 1. Load Data
train_df = pd.read_csv(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\train.csv", parse_dates=['date'])
test_df_original = pd.read_csv(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\test.csv", parse_dates=['date'])

# 2. Run Pipeline (Xtest_final WILL NOW CONTAIN 'id')
Xtrain, Ytrain, Xtest_final, _ = run_pipeline(
    train_df, 
    test_df_original, 
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\oil.csv",
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\stores.csv", 
    r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\holidays_events.csv"
)

# 3. Train & Predict
family_models = train_family_models(Xtrain, Ytrain)
final_predictions = predict_family_models(family_models, Xtest_final)

# 4. Save Submission (Using the 'id' column we preserved)
submission = pd.DataFrame({
    'id': Xtest_final['id'].astype(int), # Use the actual ID column!
    'sales': final_predictions
})

# Kaggle requires 'id' to be sorted if you want to be safe, though usually mapping is enough.
# But let's sort by ID just in case to match sample_submission format.
submission = submission.sort_values('id')

submission.to_csv(r"C:\Users\fogat\Desktop\ML_test\store-sales-time-series-forecasting\submission.csv", index=False)
print("Predictions saved successfully.")
# %%
