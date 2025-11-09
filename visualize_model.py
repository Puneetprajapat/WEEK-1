# visualize_model.py"""

import pandas as pdGenerate visualizations for the trained model:

import numpy as np - predicted_vs_actual.png  (scatter + density)

import matplotlib.pyplot as plt - residual_hist.png        (histogram of residuals)

import seaborn as sns - feature_importances.png  (bar chart)

import joblib

This script loads the saved model and the cleaned CSV, reproduces the train/test split

def plot_feature_importance(model, feature_names):(with random_state=42) and saves the plots to the workspace.

    """Plot feature importance from the Random Forest model.""""""

    importance = model.feature_importances_import os

    indices = np.argsort(importance)[::-1]from pathlib import Path

    import joblib

    plt.figure(figsize=(10, 6))import pandas as pd

    plt.title('Feature Importance')import numpy as np

    plt.bar(range(len(importance)), importance[indices])import matplotlib

    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)matplotlib.use('Agg')

    plt.tight_layout()import matplotlib.pyplot as plt

    plt.savefig('feature_importances.png')import seaborn as sns

    plt.close()from sklearn.model_selection import train_test_split



def plot_predictions_vs_actual(y_true, y_pred):BASE = Path(__file__).resolve().parent

    """Create scatter plot of predicted vs actual values."""MODEL = BASE / 'household_rf_model.joblib'

    plt.figure(figsize=(10, 6))CSV = BASE / 'household_power_consumption_clean_sample.csv'

    plt.scatter(y_true, y_pred, alpha=0.5)

    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)print('Loading model:', MODEL)

    plt.xlabel('Actual Global Active Power')model = joblib.load(MODEL)

    plt.ylabel('Predicted Global Active Power')print('Loading CSV:', CSV)

    plt.title('Predicted vs Actual Values')df = pd.read_csv(CSV)

    plt.tight_layout()

    plt.savefig('predicted_vs_actual.png')# parse dt

    plt.close()if 'Datetime' in df.columns:

    df['dt'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')

def plot_residuals(y_true, y_pred):else:

    """Plot histogram of residuals."""    df['dt'] = pd.to_datetime(df['Date'].str.strip() + ' ' + df['Time'].str.strip(), dayfirst=True, errors='coerce')

    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))# engineer time features

    plt.hist(residuals, bins=50)if 'hour' not in df.columns:

    plt.xlabel('Residual Value')    df['hour'] = df['dt'].dt.hour

    plt.ylabel('Count')    df['minute'] = df['dt'].dt.minute

    plt.title('Histogram of Residuals')    df['weekday'] = df['dt'].dt.weekday

    plt.tight_layout()

    plt.savefig('residual_hist.png')features = ['Global_reactive_power', 'Voltage', 'Global_intensity',

    plt.close()            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',

            'hour', 'minute', 'weekday']

def visualize_model():features = [f for f in features if f in df.columns]

    """Create visualizations for model analysis."""print('Features used:', features)

    # Load test data

    print("Loading test data...")target = 'Global_active_power'

    test_data = pd.read_csv('predictions_sample.csv')# drop NA

    df = df.dropna(subset=[target, 'dt'])

    # Separate features and targetX = df[features]

    target = 'Global_active_power'y = df[target]

    features = [col for col in test_data.columns if col != target]

    X_test = test_data[features]# same split

    y_test = test_data[target]X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Test size:', X_test.shape)

    # Load model

    print("Loading model...")# predict

    model = joblib.load('household_rf_model.joblib')print('Predicting on test set...')

    y_pred = model.predict(X_test)

    # Generate predictionsresiduals = y_test.values - y_pred

    print("Generating predictions...")

    y_pred = model.predict(X_test)# 1) predicted vs actual scatter

    plt.figure(figsize=(6,6))

    # Create visualizationsplt.scatter(y_test, y_pred, s=2, alpha=0.3)

    print("Creating visualizations...")lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]

    plot_feature_importance(model, features)plt.plot(lims, lims, 'r--', linewidth=1)

    plot_predictions_vs_actual(y_test, y_pred)plt.xlabel('Actual Global_active_power')

    plot_residuals(y_test, y_pred)plt.ylabel('Predicted Global_active_power')

    plt.title('Predicted vs Actual')

    print("Visualizations have been saved as PNG files.")plt.tight_layout()

out1 = BASE / 'predicted_vs_actual.png'

if __name__ == "__main__":plt.savefig(out1, dpi=150)

    visualize_model()plt.close()
print('Saved', out1)

# 2) residual histogram
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=100, kde=True)
plt.xlabel('Residual (y_true - y_pred)')
plt.title('Residual Distribution')
plt.tight_layout()
out2 = BASE / 'residual_hist.png'
plt.savefig(out2, dpi=150)
plt.close()
print('Saved', out2)

# 3) feature importances
# try to access underlying rf
try:
    rf = model.named_steps['rf']
    importances = rf.feature_importances_
    fi = pd.Series(importances, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(x=fi.values, y=fi.index)
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    out3 = BASE / 'feature_importances.png'
    plt.savefig(out3, dpi=150)
    plt.close()
    print('Saved', out3)
except Exception as e:
    print('Could not extract feature importances:', e)

print('All plots generated.')
