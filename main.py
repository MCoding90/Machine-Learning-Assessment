import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from matplotlib.dates import date2num
from pmdarima import auto_arima
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Define chosen tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']


# Data Retrieval
def download_stock_data(tickers):
    global stock_data
    stock_data = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        stock = yf.Ticker(ticker)
        stock_data[ticker] = stock.history(period="1y")
    return stock_data


# Preprocess data
def preprocess_data(stock_data):
    print(f"Starting processing for {tickers}")
    global closing_prices
    closing_prices = pd.DataFrame({ticker: data['Close'] for ticker, data in stock_data.items()})
    closing_prices_filled = closing_prices.ffill().bfill()
    scaler = StandardScaler()
    global scaled_data
    scaled_data = scaler.fit_transform(closing_prices_filled.T)
    print(f"Finished")
    return closing_prices_filled, closing_prices, scaled_data


# PCA and K-Means Clustering
def perform_pca_kmeans(scaled_data, closing_prices):
    # Convert the scaled data to a DataFrame with ticker symbols as the index
    scaled_df = pd.DataFrame(scaled_data, index=closing_prices.columns, columns=closing_prices.index)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=min(scaled_df.shape[0], scaled_df.shape[1]))
    reduced_data = pca.fit_transform(scaled_df)

    # Apply K-Means clustering to the reduced data
    kmeans = KMeans(n_clusters=4, random_state=0)
    clusters = kmeans.fit_predict(reduced_data)

    # Create a DataFrame with Ticker and Cluster columns
    clustered_stocks = pd.DataFrame({'Ticker': scaled_df.index, 'Cluster': clusters})

    # Additional details
    cluster_centers = kmeans.cluster_centers_
    cluster_sizes = clustered_stocks['Cluster'].value_counts()

    return clustered_stocks, cluster_centers, cluster_sizes


# Correlation Analysis
def calculate_correlations(closing_prices_filled):
    correlation_matrix = closing_prices_filled.corr()

    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    # plt.show()
    st.pyplot(plt)

    correlations = correlation_matrix.unstack()
    sorted_correlations = correlations.sort_values(kind="quicksort", ascending=False)
    top_positive = sorted_correlations[sorted_correlations < 1][:10]
    negative_correlations = sorted_correlations[sorted_correlations < 0][-10:]
    if not negative_correlations.empty:
        top_negative = negative_correlations[:10]
    else:
        top_negative = "No negative correlations found"

    return top_positive, top_negative


# Function to plot time series and histogram for a stock
def plot_stock_eda(closing_prices_filled, ticker):
    data = closing_prices_filled[ticker]  # Use the processed data

    # Time Series Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data)
    plt.title(f'Time Series - Closing Price of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data, kde=True)
    plt.title(f'Histogram - Closing Price of {ticker}')
    plt.xlabel('Closing Price')
    plt.ylabel('Frequency')

    plt.tight_layout()

    st.pyplot(plt)


def arima_forecasting(stock_data, ticker, forecast_days=30):
    if ticker not in stock_data.columns:
        raise ValueError(f"Ticker {ticker} not found in data.")

    # Select stock data
    chosen_stock = stock_data[ticker].dropna()

    # Ensure there's enough data for forecasting
    if len(chosen_stock) < (2 * forecast_days):
        raise ValueError(f"Not enough data for forecasting {forecast_days} days. Only {len(chosen_stock)} data points "
                         f"available.")

    # Fit auto_arima model
    model = auto_arima(chosen_stock, start_p=1, start_q=1,
                       max_p=3, max_q=3, m=12,
                       start_P=0, seasonal=True,
                       d=1, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    # Forecast
    forecast, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)

    # Dates for plotting forecast
    last_date = chosen_stock.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='B')[1:]  # 'B' for business days

    # Plot Forecast
    plt.figure(figsize=(10, 5))
    plt.plot(chosen_stock.index, chosen_stock, label='Actual')
    plt.plot(forecast_dates, forecast, color='red', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'ARIMA Forecast for {ticker}')
    plt.legend()

    st.pyplot(plt)

    return forecast, conf_int, model


# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


# LSTM Stock Prediction
def lstm_forecasting(closing_prices, ticker, time_step=10, epochs=50, batch_size=32):
    # Extract the stock data for the chosen ticker
    stock_data = closing_prices[ticker].values.reshape(-1, 1)

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stock_data)

    # Creating training and test data
    train_size = int(len(scaled_data) * 0.65)
    train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data), :]

    # Prepare data for LSTM
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile and fit model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Prediction
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Flatten predictions
    train_predict = train_predict.flatten()
    test_predict = test_predict.flatten()

    # Adjust the indices to align with the original dataset
    train_predict_indices = np.arange(time_step, time_step + len(train_predict))
    test_predict_indices = np.arange(len(scaled_data) - len(test_predict), len(scaled_data))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
    plt.plot(train_predict_indices, train_predict, label='Train Predict')
    plt.plot(test_predict_indices, test_predict, label='Test Predict')
    plt.title(f'Stock Price Prediction using LSTM for {ticker}')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)


# Function to prepare time series data for Random forest
def create_lagged_features(df, lags=[1, 2, 3, 4, 5]):
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df.dropna()


# Random Forest for Stock Prediction
def random_forest_stock_prediction(closing_prices, ticker, n_estimators=100, test_size=0.2):
    # Check if ticker is in the DataFrame columns
    if ticker not in closing_prices.columns:
        st.error(f"Ticker {ticker} not found in the data.")
        return

    # Extract the series for the selected ticker
    stock_series = closing_prices[ticker]

    # Ensure the series is not empty
    if stock_series.empty:
        st.error(f"No data available for ticker {ticker}.")
        return
    df = pd.DataFrame({'y': closing_prices[ticker]})
    lagged_df = create_lagged_features(df)

    X = lagged_df.drop('y', axis=1)
    y = lagged_df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)

    train_preds = rf_model.predict(X_train)
    test_preds = rf_model.predict(X_test)

    if not y_train.empty and not y_test.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(y_train, label='Training Data')
        plt.plot(y_test.index, test_preds, label='Random Forest Prediction')
        plt.title(f'Random Forest Stock Prediction for {ticker}')
        plt.xlabel('Time Step')
        plt.ylabel('Stock Price')
        plt.legend()
        st.pyplot(plt)
    else:
        st.error('Error in data for plotting.')

    return train_preds, test_preds, y_train, y_test


# Linear Regression
def linear_regression_prediction(stock_price, test_size=0.30):
    # Split data into training and testing set
    train, test = train_test_split(stock_price.to_frame(), test_size=test_size)

    # Reshape x and y training data
    x_train = date2num(train.index).astype(float).reshape(-1, 1)
    y_train = train[stock_price.name]  # Use the name of the stock_price Series as the target column name

    # Create Model and fit the data
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Linear Regression predictions VS the actual price
    plt.figure(figsize=(12, 6))
    plt.title(f"Linear Regression Prediction for {stock_price.name}")
    plt.scatter(x_train, y_train, edgecolors='w', label='Actual Price')
    plt.plot(x_train, model.predict(x_train), color='r', label='Predicted Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)


# Function to get the current price of a stock
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="1d")["Close"].iloc[-1]
    return current_price


# Function to generate ARIMA forecast and trading signals
def generate_arima_signals(stock_data, ticker, forecast_days, threshold=0.05):
    try:
        if ticker not in stock_data.columns:
            raise ValueError(f"Ticker {ticker} not found in DataFrame columns: {stock_data.columns}")

        current_price = stock_data[ticker].iloc[-1]

        model = auto_arima(stock_data[ticker], seasonal=False, error_action='ignore', suppress_warnings=True)
        forecast, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)

        # Use the last forecast for signal generation
        forecast_price = forecast.iloc[-1]

        change_percent = (forecast_price - current_price) / current_price

        if change_percent > threshold:
            signal = 'Buy'
        elif change_percent < -threshold:
            signal = 'Sell'
        else:
            signal = 'Hold'

        return {
            'current_price': current_price,
            'forecast_price': forecast_price,
            'change_percent': change_percent,
            'signal': signal
        }

    except Exception as e:
        print(f"Error in generate_arima_signals: {repr(e)}")
        raise


def main():
    st.title("Stock Market Analysis and Prediction by Maciej Czerwonka")

    # Initialize session state
    if 'closing_prices_filled' not in st.session_state:
        st.session_state['closing_prices_filled'] = pd.DataFrame()

    selected_stock = st.sidebar.selectbox("Select a stock for analysis", tickers)
    forecast_period = st.sidebar.selectbox("Select Forecast Period", [7, 14, 30])

    if st.sidebar.button('Load Data'):
        with st.spinner('Downloading and processing data...'):
            stock_data = download_stock_data(tickers)
            processed_data = preprocess_data(stock_data)
            st.session_state['closing_prices_filled'] = processed_data[0] if processed_data else pd.DataFrame()
            st.session_state['closing_prices'] = closing_prices
            st.session_state['scaled_data'] = scaled_data
            st.success("Data Loaded Successfully!")

    if st.sidebar.checkbox("Show Stock EDA"):
        if not st.session_state['closing_prices_filled'].empty:
            plot_stock_eda(st.session_state['closing_prices_filled'], selected_stock)
        else:
            st.error("Data not loaded. Please load data first.")

    # PCA and K-Means Clustering
    if st.sidebar.checkbox("Perform PCA and KMeans Clustering"):
        if 'scaled_data' in st.session_state and 'closing_prices' in st.session_state:
            # Pass both scaled_data and closing_prices to the function
            clustered_stocks, cluster_centers, cluster_sizes = perform_pca_kmeans(
                st.session_state['scaled_data'], st.session_state['closing_prices']
            )

            st.write("PCA and KMeans Clustering Results:")
            st.dataframe(clustered_stocks)

            st.write("Cluster Centers:")
            st.dataframe(cluster_centers)

            st.write("Cluster Sizes:")
            st.dataframe(cluster_sizes)
        else:
            st.error("Data not loaded or preprocessed. Please load and preprocess data first.")

    # Correlation Analysis
    if st.sidebar.checkbox("Show Correlation Analysis"):
        if not st.session_state['closing_prices_filled'].empty:
            top_positive, top_negative = calculate_correlations(st.session_state['closing_prices_filled'])
            st.write("Top Positive Correlations:", top_positive)
            st.write("Top Negative Correlations:", top_negative)
        else:
            st.error("Data not loaded. Please load data first.")

    # ARIMA Forecasting
    if st.sidebar.checkbox("Perform ARIMA Forecasting"):
        if 'closing_prices_filled' in st.session_state:
            forecast_days = st.sidebar.number_input("Select Number of Forecast Days", 10, 60, 30)
            forecast, conf_int, model = arima_forecasting(st.session_state['closing_prices_filled'], selected_stock,
                                                          forecast_days)
            st.write(forecast)
        else:
            st.error("Data not loaded. Please load data first.")

    # LSTM Forecasting
    if st.sidebar.checkbox("Perform LSTM Forecasting"):
        if 'closing_prices_filled' in st.session_state:
            lstm_forecasting(st.session_state['closing_prices_filled'], selected_stock)
        else:
            st.error("Data not loaded. Please load data first.")

    # Linear Regression
    if st.sidebar.checkbox("Perform Linear Regression"):
        if 'closing_prices_filled' in st.session_state:
            stock_price_series = st.session_state['closing_prices_filled'][selected_stock]
            linear_regression_prediction(stock_price_series)
        else:
            st.error("Data not loaded. Please load data first.")

    # Random Forest Prediction
    if st.sidebar.checkbox("Perform Random Forest Prediction"):
        if selected_stock in st.session_state['closing_prices_filled'].columns:
            random_forest_stock_prediction(st.session_state['closing_prices_filled'], selected_stock)
        else:
            st.error(f"Ticker {selected_stock} not found in the data.")

    if st.button('Generate Signal'):
        try:
            # Check if data is loaded
            if 'closing_prices_filled' not in st.session_state or st.session_state['closing_prices_filled'].empty:
                st.error("Data not loaded. Please load data first.")
                return

            # Set 'Date' as index if it is in the columns
            if 'Date' in st.session_state['closing_prices_filled'].columns:
                st.session_state['closing_prices_filled'].set_index('Date', inplace=True)

            # Additional check before calling the function
            if selected_stock not in st.session_state['closing_prices_filled'].columns:
                st.error(f"Selected stock {selected_stock} not found in the DataFrame.")
                return

            signal_info = generate_arima_signals(st.session_state['closing_prices_filled'], selected_stock,
                                                 forecast_period)
            st.write(
                f"Trading signal using ARIMA for {selected_stock} for {forecast_period} days: {signal_info['signal']}")
            st.write(f"Current Price: {signal_info['current_price']}, Forecast Price: {signal_info['forecast_price']}")

        except Exception as e:
            st.error(f"Error occurred: {repr(e)}")


if __name__ == "__main__":
    main()
