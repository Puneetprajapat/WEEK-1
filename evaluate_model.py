# evaluate_model.pyimport os

import pandas as pdfrom pathlib import Path

import numpy as npimport joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_scoreimport pandas as pd

import joblibimport numpy as np

from sklearn.model_selection import train_test_split

def load_model(model_path):from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    """Load the trained model."""

    return joblib.load(model_path)BASE = Path(__file__).resolve().parent

# find model file

def evaluate_predictions(y_true, y_pred):model_path = BASE / 'household_rf_model.joblib'

    """Calculate and return evaluation metrics."""if not model_path.exists():

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))    # try other names

    mae = mean_absolute_error(y_true, y_pred)    for p in BASE.glob('*.joblib'):

    r2 = r2_score(y_true, y_pred)        model_path = p

    return rmse, mae, r2        break



def evaluate_model():if not model_path.exists():

    """Evaluate the model on test data."""    print('No model file found in', BASE)

    # Load test data    raise SystemExit(1)

    print("Loading test data...")

    test_data = pd.read_csv('predictions_sample.csv')print('Model file:', model_path)

    print('Size (MB):', round(os.path.getsize(model_path)/1024/1024, 3))

    # Separate features and target

    target = 'Global_active_power'# Load model

    features = [col for col in test_data.columns if col != target]model = joblib.load(model_path)

    X_test = test_data[features]print('Loaded model type:', type(model))

    y_test = test_data[target]

    # Locate csv

    # Load modelpossible = [

    print("Loading model...")    'household_power_consumption_clean_sample_trim1.csv',

    model = load_model('household_rf_model.joblib')    'household_power_consumption_clean_sample.csv',

        'household_power_consumption_clean_sample_trim2.csv'

    # Make predictions]

    print("Making predictions...")for fname in possible:

    y_pred = model.predict(X_test)    csv = BASE / fname

        if csv.exists():

    # Calculate metrics        break

    rmse, mae, r2 = evaluate_predictions(y_test, y_pred)else:

        csv = BASE / 'household_power_consumption_clean_sample.csv'

    # Print results

    print("\nModel Evaluation Results:")if not csv.exists():

    print(f"Root Mean Square Error: {rmse:.4f}")    print('No CSV found at', csv)

    print(f"Mean Absolute Error: {mae:.4f}")    raise SystemExit(1)

    print(f"R-squared Score: {r2:.4f}")

    print('Using CSV:', csv)

    return rmse, mae, r2

df = pd.read_csv(csv)

if __name__ == "__main__":# parse dt like train script

    evaluate_model()if 'Datetime' in df.columns:
    df['dt'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')
else:
    df['dt'] = pd.to_datetime(df['Date'].str.strip() + ' ' + df['Time'].str.strip(), dayfirst=True, errors='coerce')

features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'hour', 'minute', 'weekday']
# create time features
if 'hour' not in df.columns:
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['second'] = df['dt'].dt.second
    df['weekday'] = df['dt'].dt.weekday

features = [f for f in features if f in df.columns]
print('Features used:', features)

target = 'Global_active_power'

# drop rows missing target or dt
df = df.dropna(subset=[target, 'dt'])
X = df[features]
y = df[target]

# create same train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train/Test shapes:', X_train.shape, X_test.shape)

# predict
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nEvaluation on test split:')
print(f'RMSE: {rmse:.4f}')
print(f'MAE:  {mae:.4f}')
print(f'R2:   {r2:.4f}')

# save a small sample of predictions
out_sample = BASE / 'predictions_sample.csv'
sample_df = X_test.copy().head(20)
sample_df['y_true'] = y_test.head(20).values
sample_df['y_pred'] = y_pred[:20]
sample_df.to_csv(out_sample, index=False)
print('Wrote prediction sample to', out_sample)
