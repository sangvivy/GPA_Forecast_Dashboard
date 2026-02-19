
# app.py
import streamlit as st
st.set_page_config(layout="wide", page_title="GPA Forecast Dashboard")
st.write("🔥 APP STARTED SUCCESSFULLY")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import math
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

#uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

#if uploaded_file is not None:
#    df_raw = pd.read_csv(uploaded_file)
#    df = df_raw.copy()
#    st.success("Dataset loaded successfully!")
#else:
#    st.warning("Please upload a CSV file to continue.")


# Optional auto_arima
try:
    from pmdarima import auto_arima
    HAS_PMD = True
except Exception:
    HAS_PMD = False




st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
        📊 GPA Forecast Dashboard
    </h1>
    <hr style='border:1px solid #4CAF50'>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("GPA Forecast Dashboard — ARIMA / SARIMA / Holt-Winters")
#st.markdown(
  #  "Upload a CSV with columns: `studentID`, `Year`, `Semester` (Spring/Summer/Fall), `GPA`."
#)





# -------------------------
# Helpers
# -------------------------
semester_to_month = {'spring': 3, 'summer': 7, 'fall': 11}

def safe_metrics(true, pred):
    true_vals = np.array(true)
    pred_vals = np.array(pred)
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100
    return {'rmse': rmse, 'mae': mae, 'mape': mape}

def preprocess_df(df):
    # Normalize column names
    df = df.rename(columns=lambda x: x.strip())
    required = {'studentID', 'Year', 'Semester', 'GPA'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    df = df[['studentID', 'Year', 'Semester', 'GPA']].copy()
    df['Semester'] = df['Semester'].astype(str).str.strip().str.lower()
    # map Semester to month
    df['month'] = df['Semester'].map(semester_to_month)
    # If mapping failed for some rows, raise error
    if df['month'].isna().any():
        bad = df.loc[df['month'].isna(), 'Semester'].unique()
        raise ValueError(f"Unknown semester names found: {bad}. Use Spring, Summer, Fall.")
    df['Year'] = df['Year'].astype(int)
    df['date'] = pd.to_datetime(dict(year=df['Year'], month=df['month'], day=1))
    df = df.sort_values(['studentID', 'date'])
    return df

def build_student_ts(df_student):
    # Produces a regular time series indexed by the semester dates
    s = df_student.set_index('date')['GPA'].sort_index()
    # Ensure frequency; we use quarter-like anchor (3-month separation won't exactly match semesters but is OK)
    try:
        s = s.asfreq('QS-MAR')  # anchor to March quarters (our months are Mar, Jul, Nov -> uses 4MS inferred but okay)
    except Exception:
        # fallback: keep as-is
        s = s
    # If there are NaNs from asfreq, fill by interpolation or leave as is (models may need contiguous series)
    if s.isna().any():
        s = s.interpolate().ffill().bfill()
    return s

def safe_plot_series(ax, history, test=None, forecasts=None, labels=None, title=""):
    ax.plot(history.index, history.values, marker='o', label='History')
    if test is not None:
        ax.plot(test.index, test.values, marker='o', label='Test (held-out)')
    if forecasts:
        for name, fc in forecasts.items():
            try:
                ax.plot(fc.index, fc.values, marker='x', label=name)
            except Exception:
                pass
    ax.set_ylim(0, 4.2)
    ax.set_title(title)
    ax.legend()


# -------------------------
# UI: data upload / sample
# -------------------------
upload_col, sample_col = st.columns([1,1])
with upload_col:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], help="CSV must have studentID, Year, Semester, GPA")
with sample_col:
    use_sample = st.checkbox("Use simulated sample dataset (instead of upload)", value=False)

if uploaded_file is None and not use_sample:
    st.info("Upload a CSV or check 'Use simulated sample dataset' to try the app.")
    st.stop()

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(file_bytes)

if use_sample:
    # Create a simulated dataset (like we used previously)
    def create_sample():
        np.random.seed(42)
        students = list(range(1001, 1001+500))
        years = list(range(2020, 2026))
        semesters = ['Spring', 'Summer', 'Fall']
        rows = []
        for sid in students:
            base = np.clip(np.random.normal(3.0, 0.3), 0.0, 4.0)
            trend = np.random.normal(0.0, 0.02)
            noise_scale = np.random.uniform(0.05, 0.2)
            t = 0
            for y in years:
                for sem in semesters:
                    season_effect = {'Spring':0.03, 'Summer':-0.02, 'Fall':0.00}[sem]
                    gpa = base + trend*t + season_effect + np.random.normal(0, noise_scale)
                    gpa = np.clip(gpa, 0.0, 4.0)
                    rows.append({'studentID': sid, 'Year': y, 'Semester': sem, 'GPA': round(gpa, 3)})
                    t += 1
        return pd.DataFrame(rows)
    df_raw = create_sample()
else:
    # Load uploaded CSV
    try:
        df_raw = load_csv(uploaded_file)
        st.write("STEP CHECK ✅ CSV Loaded:", df_raw.shape)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
       
        st.stop()

df = df_raw.copy()

#df = pd.read_csv("student_gpa.csv")
# OR your simulation code
      
# Preprocess
try:
    df = preprocess_df(df_raw)
    st.write(df.head())
    st.write("STEP CHECK ✅ Preprocessed:", df.shape)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

#df = simulate_dataset()
#df = preprocess_df(df)

col1, col2, col3 = st.columns(3)

col1.metric("Average GPA", f"{df['GPA'].mean():.2f}")
col2.metric("Highest GPA", f"{df['GPA'].max():.2f}")
col3.metric("Lowest GPA", f"{df['GPA'].min():.2f}")

  
# Student selection
st.sidebar.markdown("## 🔧 Controls")

student_list = sorted(df['studentID'].unique())
selected_student = st.sidebar.selectbox("Select Student", student_list)

horizon = st.sidebar.slider("Forecast Horizon (semesters)", 1, 8, 4)



# Forecast horizon slider
horizon = st.slider("Forecast horizon (semesters)", min_value=1, max_value=8, value=3)

# -------------------------
# Extract series & split
# -------------------------
df_student = df[df['studentID'] == selected_student].copy()
if df_student.empty:
    st.error("No data for selected student.")
    st.stop()

ts = build_student_ts(df_student)

st.write(f"Observations for student {selected_student}: {len(ts)} semesters (from {ts.index.min().date()} to {ts.index.max().date()})")
st.dataframe(df_student[['Year','Semester','GPA']].reset_index(drop=True).head(12))

# require at least 6 points to attempt modeling; otherwise warn
if len(ts) < 6:
    st.warning("Series is very short (<6). Models may be unreliable. You can still try, but interpret results with caution.")

# Train/test split: hold last min(3, len-1) for test if possible
test_periods = min(3, max(1, len(ts)//6))  # default 3, but avoid if too short
train = ts.iloc[:-test_periods]
test = ts.iloc[-test_periods:]

# -------------------------
# Model fitting functions (cached)
# -------------------------
@st.cache_data
def fit_arima_model(series):
    # fallback ARIMA(1,1,1) if pmdarima not available
    if HAS_PMD and len(series) >= 10:
        try:
            auto = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', max_p=4, max_q=4)
            order = auto.order
        except Exception:
            order = (1,1,1)
    else:
        order = (1,1,1)
    model = ARIMA(series, order=order)
    res = model.fit()
    return res, order

@st.cache_data
def fit_sarima_model(series, seasonal_periods=3):
    # default seasonal order (1,1,1,s)
    order = (1,1,1)
    seasonal_order = (1,1,1,seasonal_periods)
    if HAS_PMD and len(series) >= 12:
        try:
            auto = auto_arima(series, seasonal=True, m=seasonal_periods, stepwise=True, suppress_warnings=True, error_action='ignore',
                              max_p=3, max_q=3, max_P=2, max_Q=2)
            order = auto.order
            seasonal_order = auto.seasonal_order
        except Exception:
            pass
    model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res, order, seasonal_order

@st.cache_data
def fit_hw_model(series, seasonal_periods=3, seasonal='add'):
    model = ExponentialSmoothing(series, trend='add', seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method='estimated')
    res = model.fit(optimized=True)
    return res

# -------------------------
# Fit models with spinner
# -------------------------
with st.spinner("Fitting models — this may take a few seconds..."):
    try:
        arima_res, arima_order = fit_arima_model(train)
    except Exception as e:
        st.error(f"ARIMA fit failed: {e}")
        arima_res, arima_order = None, None
    try:
        sarima_res, sar_order, sar_seasonal = fit_sarima_model(train, seasonal_periods=3)
    except Exception as e:
        st.error(f"SARIMA fit failed: {e}")
        sarima_res, sar_order, sar_seasonal = None, None, None
    try:
        hw_res = fit_hw_model(train, seasonal_periods=3, seasonal='add')
    except Exception as e:
        st.error(f"Holt-Winters fit failed: {e}")
        hw_res = None

st.success("Model fitting complete.")

# -------------------------
# Forecast & Evaluate
# -------------------------
forecasts = {}
metrics = []

# ARIMA
if arima_res is not None:
    try:
        arima_fc = arima_res.get_forecast(steps=horizon).predicted_mean
        forecasts[f"ARIMA (order={arima_order})"] = arima_fc
        # Evaluate on the held-out test (if we forecast test_periods)
        try:
            arima_pred_test = arima_res.get_forecast(steps=test_periods).predicted_mean
            m = safe_metrics(test, arima_pred_test)
        except Exception:
            m = {'rmse':np.nan, 'mae':np.nan, 'mape':np.nan}
        metrics.append({'model': 'ARIMA', **m})
    except Exception as e:
        st.warning(f"ARIMA forecast failed: {e}")
        metrics.append({'model':'ARIMA', 'rmse':np.nan, 'mae':np.nan, 'mape':np.nan})
else:
    metrics.append({'model':'ARIMA', 'rmse':np.nan, 'mae':np.nan, 'mape':np.nan})

# SARIMA
if sarima_res is not None:
    try:
        sarima_fc = sarima_res.get_forecast(steps=horizon).predicted_mean
        forecasts[f"SARIMA (s={sar_seasonal[3]})"] = sarima_fc
        try:
            sar_pred_test = sarima_res.get_forecast(steps=test_periods).predicted_mean
            m = safe_metrics(test, sar_pred_test)
        except Exception:
            m = {'rmse':np.nan, 'mae':np.nan, 'mape':np.nan}
        metrics.append({'model':'SARIMA', **m})
    except Exception as e:
        st.warning(f"SARIMA forecast failed: {e}")
        metrics.append({'model':'SARIMA', 'rmse':np.nan, 'mae':np.nan, 'mape':np.nan})
else:
    metrics.append({'model':'SARIMA', 'rmse':np.nan, 'mae':np.nan, 'mape':np.nan})

# Holt-Winters
if hw_res is not None:
    try:
        hw_fc = hw_res.forecast(horizon)
        forecasts[f"Holt-Winters (add)"] = hw_fc
        try:
            hw_pred_test = hw_res.forecast(test_periods)
            m = safe_metrics(test, hw_pred_test)
        except Exception:
            m = {'rmse':np.nan, 'mae':np.nan, 'mape':np.nan}
        metrics.append({'model':'Holt-Winters (add)', **m})
    except Exception as e:
        st.warning(f"Holt-Winters forecast failed: {e}")
        metrics.append({'model':'Holt-Winters (add)', 'rmse':np.nan, 'mae':np.nan, 'mape':np.nan})
else:
    metrics.append({'model':'Holt-Winters (add)', 'rmse':np.nan, 'mae':np.nan, 'mape':np.nan})

metrics_df = pd.DataFrame(metrics)

# -------------------------
# Display: plots and metrics
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    fig, ax = plt.subplots(figsize=(10,5))
    # plot history and held-out test
    safe_plot_series(ax, history=ts, test=test, forecasts=forecasts, title=f"Student {selected_student} - Forecasts (h={horizon})")
    st.pyplot(fig)

with col2:
    st.subheader("Model metrics (on held-out test)")
    # format display
    display_df = metrics_df.copy()
    display_df['rmse'] = display_df['rmse'].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    display_df['mae'] = display_df['mae'].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    display_df['mape'] = display_df['mape'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "NA")
    st.table(display_df.set_index('model'))

    # choose best by rmse
    try:
        best_row = metrics_df.loc[metrics_df['rmse'].idxmin()]
        st.success(f"Best model by RMSE: {best_row['model']} (RMSE={best_row['rmse']:.4f})")
    except Exception:
        st.info("Best model not available (insufficient metrics).")

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Student Trend", "📘 Dataset"])
with tab1:
    st.subheader("GP‌A Forecast")
    st.line_chart(forecasts)
with tab2:
    st.subheader("Student GPA Trend")
    st.line_chart(ts)
with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(df)

# -------------------------
# Forecast table & download
# -------------------------
st.subheader("Forecast table (all models)")
out_df = []
for name, ser in forecasts.items():
    # ensure index exists, otherwise create future index
    if ser.index is None or len(ser.index)==0:
        # create future dates spaced by 4 months starting after last date
        start = ts.index[-1] + pd.DateOffset(months=4)
        ser.index = pd.date_range(start=start, periods=len(ser), freq='QS-MAR')
    tmp = pd.DataFrame({'date': ser.index, 'model': name, 'forecast': ser.values})
    out_df.append(tmp)
if out_df:
    out_df = pd.concat(out_df).pivot(index='date', columns='model', values='forecast').reset_index()
    st.dataframe(out_df)
    # download
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download forecasts CSV", data=csv, file_name=f"forecasts_student_{selected_student}.csv", mime='text/csv')
else:
    st.info("No forecasts to show.")

st.markdown("---")
st.caption("Note: With short series (few semesters) model results may be unstable. ARIMA/SARIMA perform better with more observations.")
