import emoji
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vizro.models.types import capture
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from ntscraper import Nitter
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as sia
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from datetime import datetime, timedelta
import requests
import os
# Define model name and directory for storing models
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
# Gets the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Subdirectory 'models' within script's directory
local_model_dir = os.path.join(script_directory, 'models')

# Function to check if model & tokenizer are downloaded
def check_model_downloaded(directory, model_name):
    required_files = ["pytorch_model.bin", "config.json",
                      "tokenizer_config.json", "vocab.json", "merges.txt"]
    return all(os.path.exists(os.path.join(directory, file)) for file in required_files)


# Ensure the local model directory exists
if not os.path.exists(local_model_dir):
    os.makedirs(local_model_dir)

# Check if model and tokenizer are downloaded
if not check_model_downloaded(local_model_dir, model_name):
    # Download and save the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    print("Model and tokenizer downloaded and saved.")
else:
    print("Model and tokenizer already downloaded.")


nltk.download('words')
nltk.download('vader_lexicon')

import warnings
warnings.filterwarnings('ignore')

# ------------------------------ Strategies ------------------------------ #

def get_linear_regression_predictions(data):
    predictions = {}
    for price_type in ['Open', 'High', 'Low', 'Close']:
        temp_data = data.copy()
        temp_data['Prediction'] = temp_data[price_type].shift(-1)
        last_row = temp_data.iloc[[-1]].drop(['Prediction', 'Symbol'], axis=1)
        temp_data.dropna(inplace=True)
        X = np.array(temp_data.drop(['Prediction', 'Symbol'], axis=1))
        Y = np.array(temp_data['Prediction'])

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2)

        model = LinearRegression()
        model.fit(x_train, y_train)

        predictions_train = model.predict(X)

        prediction_last_row = model.predict(last_row)

        all_predictions = np.concatenate(
            [predictions_train, prediction_last_row])

        predictions[price_type] = all_predictions

    return pd.DataFrame(predictions)


def get_ridge_lasso_regression_predictions(data, model_type='ridge'):
    predictions = {}
    scaler = StandardScaler()
    param_grid = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100]}

    for price_type in ['Open', 'High', 'Low', 'Close']:
        temp_data = data.copy()
        temp_data['Prediction'] = temp_data[price_type].shift(-1)
        last_row = temp_data.iloc[[-1]].drop(['Prediction', 'Symbol'], axis=1)
        temp_data.dropna(inplace=True)
        X = np.array(temp_data.drop(['Prediction', 'Symbol'], axis=1))
        Y = np.array(temp_data['Prediction'])
        X_scaled = scaler.fit_transform(X)
        last_row_scaled = scaler.transform(last_row)

        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, Y, test_size=0.2)

        if model_type == 'ridge':
            model = Ridge()
        else:
            model = Lasso()

        grid = GridSearchCV(model, param_grid,
                            scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_

        predictions_train = best_model.predict(X_scaled)

        prediction_last_row = best_model.predict(last_row_scaled)

        all_predictions = np.concatenate(
            [predictions_train, prediction_last_row])

        predictions[price_type] = all_predictions

    return pd.DataFrame(predictions)


def get_ridge_lasso_combo_predictions(data):

    lasso_predictions = get_ridge_lasso_regression_predictions(data, 'lasso')

    ridge_predictions = get_ridge_lasso_regression_predictions(data, 'ridge')
    combined_predictions = {
        'Open': lasso_predictions['Open'],
        'Low': lasso_predictions['Low'],   
        'High': ridge_predictions['High'],
        'Close': ridge_predictions['Close']
    }

    return pd.DataFrame(combined_predictions)
# ------------------------------------------------------------------------------ #


# ------------------------------ Accuracy Metrics ------------------------------ #

def calculate_color_prediction_accuracy(data, predictions):
    def get_color(open_price, close_price):
        return 'green' if open_price <= close_price else 'red'

    actual_colors = [get_color(row['Open'], row['Close'])
                     for _, row in data.iterrows()]

    predicted_colors = [get_color(row['Open'], row['Close'])
                        for _, row in predictions.iterrows()]
    
    correct_predictions = sum(a == p for a, p in zip(
        actual_colors, predicted_colors))
    total_predictions = len(actual_colors)
    accuracy = correct_predictions / total_predictions

    return accuracy


def calculate_volatility_prediction_score(data, predictions):
    def get_volatility(high, low):
        return abs(high - low)

    actual_volatilities = [get_volatility(row['High'], row['Low'])
                           for _, row in data.iterrows()]

    predicted_volatilities = [get_volatility(row['High'], row['Low'])
                              for _, row in predictions.iterrows()]

    # Normalize the volatilities to a range of 0-1 for scoring
    max_actual_volatility = max(actual_volatilities)
    normalized_actual_vol = [
        v / max_actual_volatility for v in actual_volatilities]
    normalized_predicted_vol = [
        v / max_actual_volatility for v in predicted_volatilities]

    # Calculate the score
    volatility_scores = [
        1 - abs(a - p) for a, p in zip(normalized_actual_vol, normalized_predicted_vol)]
    average_score = sum(volatility_scores) / len(volatility_scores)

    return average_score


def calculate_model_accuracy(true_values, predicted_values, file_name):

    metrics = {}
    for label in ['Open', 'High', 'Low', 'Close']:
        true = true_values[label]
        predicted = predicted_values[label]

        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, predicted)
        explained_variance = explained_variance_score(true, predicted)

        metrics[label] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Explained Variance': explained_variance
        }

    color_accuracy = calculate_color_prediction_accuracy(
        true_values, predicted_values)
    volatility_score = calculate_volatility_prediction_score(
        true_values, predicted_values)

    metrics['Color Accuracy'] = {'Color Accuracy': color_accuracy}
    metrics['Volatility Score'] = {'Volatility Score': volatility_score}

    metrics_df = pd.DataFrame(metrics).T  # Transpose to get labels as rows

    metrics_df.to_csv(file_name, index_label='Label')

    # for label, metric_values in metrics.items():
    #     print(f"{label} Metrics: {metric_values}")

# ------------------------------------------------------------------------------ #

# ------------------------------ Features ------------------------------ #

# Carson's features

# VOLATILITY
def calculate_volatility(df, window=30):
    stockData = df.copy()
    # Calculate daily returns
    stockData['Daily Returns'] = stockData['Adj Close'].pct_change()

    # Calculate rolling volatility (standard deviation of daily returns)
    # The window size can be adjusted. A window of 1 will calculate the standard deviation of the daily return itself.
    stockData['Daily Volatility'] = stockData['Daily Returns'].rolling(
        window=window).std()

    return stockData['Daily Volatility']

# MOMENTUM
def calculate_momentum(df, window):
    stockData = df.copy()
    # Calculate daily returns
    stockData['Daily Returns'] = stockData['Adj Close'].pct_change()

    # Calculate momentum using a rolling window
    stockData['Momentum'] = stockData['Adj Close'].diff(window)

    return stockData['Momentum'].dropna()

    
# Ashton's features
    
def get_tweet_sentiments(ticker):
    scraper = Nitter()
    tweets = scraper.get_tweets(ticker, mode='term', number=100, language='en')
    scraped_tweet_list = []


    for tweet in tweets['tweets']:
        data = [tweet['date'], tweet['text'], tweet['stats']
                ['likes'], tweet['stats']['comments']]
        scraped_tweet_list.append(data)

    df = pd.DataFrame(scraped_tweet_list, columns=[
                      'date', 'text', 'Num_Likes', 'Num_Comments'])


    df = df.drop(columns=['date', 'Num_Likes', 'Num_Comments']) # maybe could weight number of likes/comments on sentiment

    def clean_tweets(tweet):
        tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        tweet = " ".join(tweet.split())
        tweet = ''.join(c for c in tweet if c not in emoji.EMOJI_DATA) #Remove Emojis
        tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        return tweet
    
    df['text'] = df['text'].map(lambda x: clean_tweets(x))

    analyzer = sia()


    vader_results = []
    for i, row in tqdm(df.iterrows()):
        text = row['text']
        vader_results.append(analyzer.polarity_scores(text))

    vader_res_df = pd.DataFrame(vader_results)


    # using roberta reinforcement
    model = f"cardiffnlp/twitter-roberta-base-sentiment"


    # Load the model and tokenizer from the local directory
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
 

    def roberta_scorer(tweet_example):    
        encoded = tokenizer(tweet_example, return_tensors='pt')
        output = model(**encoded)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        scores_dict = {
            'roberta-neg' : scores[0],
            'roberta-neu' : scores[1],
            'roberta-pos' : scores[2]
        }

        return scores_dict

    roberta_results = []


    for i, row in tqdm(df.iterrows()):
        text = row['text']
        roberta_results.append(roberta_scorer(text))

    roberta_res_df = pd.DataFrame(roberta_results)

    combined_df = pd.concat([vader_res_df, roberta_res_df], axis=1)

    return combined_df


def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()


def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()


def DEMA(EMA):
    return 2*EMA - EMA(EMA)


def MACD(df, long_period=26, short_period=12, signal_line_period=9, column='Close'):
    data = df.copy()
    short_EMA = EMA(data, short_period, column=column)
    long_EMA = EMA(data, long_period, column=column)

    data['MACD'] = short_EMA - long_EMA
    data['Signal Line'] = EMA(data, signal_line_period, column='MACD')

    return data


def RSI(data, period=14, column='Close'):
    delta = data[column].diff(1)
    delta = delta[1:]

    increasing = delta.copy()
    decreasing = delta.copy()
    increasing[increasing < 0] = 0
    decreasing[decreasing > 0] = 0

    data['Increasing'] = increasing
    data['Decreasing'] = decreasing

    average_gain = SMA(data, period, column='Increasing')
    average_loss = abs(SMA(data, period, column='Decreasing'))

    relative_strength = average_gain / average_loss
    RSI = 100 - (100/(1 + relative_strength))

    data['RSI'] = RSI

    return data


# Riley's features

def get_sentiments(df, ticker='AAPL'):


    URL = "https://data.alpaca.markets/v1beta1/news"

    # Set the API key and secret
    API_KEYS = {
        "APCA-API-KEY-ID": "PKHL949M71P5B43DNVKX",
        "APCA-API-SECRET-KEY": "tI2AoMbdQD7XjCEMIVsRrDi8gVW6vTYWkku8MsT2"
    }


    def get_news(ticker, start_date, end_date):

        start_date = datetime.fromisoformat(
            start_date).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = datetime.fromisoformat(end_date).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "symbols": ticker,
            "start": start_date,
            "end": end_date,
        }

        response = requests.get(URL, headers=API_KEYS, params=params)

        news_data = response.json()
        try:
            return news_data['news']
        except KeyError:
            return 0

    # Extract dates from stock data
    dates_in_stock_data = [datetime.strptime(
        date, "%Y-%m-%d").date() for date in df['Date']]

    sentiments = []
    analyzer = sia()

    for i, current_date in enumerate(dates_in_stock_data):
        day_beginning = current_date

        # Check if this is the last date in stock_data, if not then get the next trading day
        next_trading_day = dates_in_stock_data[i +
                                               1] if i+1 < len(dates_in_stock_data) else None

        # If this isn't the last date and there's a gap in trading days, combine sentiment scores
        if next_trading_day and (next_trading_day - current_date).days > 1:
            day_end = next_trading_day
        else:
            day_end = current_date + timedelta(days=1)

        tsla_news = get_news(ticker, start_date=str(
            day_beginning), end_date=str(day_end))

        if tsla_news != 0:
            articles = [article['headline'] for article in tsla_news]
            scores = [analyzer.polarity_scores(
                article)['compound'] for article in articles]
            sentiments.append(np.mean(scores))
        else:
            sentiments.append(0)  # neutral sentiment for no news

    return sentiments

def add_features(df):
    features_file = 'features_data.csv'

    # Check if the features file already exists
    if os.path.exists(features_file):
        print(f"'{features_file}' exists. Loading features from the file.")
        return pd.read_csv(features_file).dropna()

    df['30 day volatility'] = calculate_volatility(df, 30)
    df['30 day momentum'] = calculate_momentum(df, 30)
    df['5 day volatility'] = calculate_volatility(df, 5)
    df['5 day momentum'] = calculate_momentum(df, 5)

    print("getting tweets sentiment")
    sentiments = get_tweet_sentiments(df['Symbol'].iloc[0])
    df = pd.concat([df, sentiments], axis=1)

    MACD(df)
    RSI(df)
    df['SMA'] = SMA(df)
    df['EMA'] = EMA(df)
    #print("getting news sentiments")
   # df['news sentiment'] = get_sentiments(df, ticker=df['Symbol'].iloc[0])

    df.to_csv(features_file)

    return df.dropna()


# ----------------------------------- Graphs ----------------------------------- #

def plot_util(true, method, metrics_file_name):
    time = true['Date']
    true = true.drop(columns=['Date', 'DateTime'])

    if 'ridge' in metrics_file_name:
        predictions = method(true, 'ridge')
    elif 'lasso' in metrics_file_name:
        predictions = method(true, 'lasso')
    else:
        predictions = method(true)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=time,
                                 open=true['Open'],
                                 high=true['High'],
                                 low=true['Low'],
                                 close=true['Close'],
                                 name='True',
                                 increasing_line_color='green',
                                 decreasing_line_color='red'))

    fig.add_trace(go.Candlestick(x=time,
                                 open=predictions['Open'],
                                 high=predictions['High'],
                                 low=predictions['Low'],
                                 close=predictions['Close'],
                                 name='Predicted',
                                 increasing_line_color='lightgreen',
                                 decreasing_line_color='lightcoral'))

    split_index = int(len(true) * 0.8)

    fig.add_vrect(
        x0=time.iloc[0], x1=time.iloc[split_index],
        fillcolor="blue", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Training Data", annotation_position="top left"
    )

    fig.add_vrect(
        x0=time.iloc[split_index], x1=time.iloc[-1],
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Testing Data", annotation_position="top right"
    )

    fig.update_layout(xaxis_rangeslider_visible=False,
                      title="True vs Predicted Candlestick Chart", xaxis_title="Date", yaxis_title="Price")

    test_df = true.iloc[int(len(true) * 0.8):]
    test_predictions = predictions.iloc[int(
        len(predictions) * 0.8):]
    calculate_model_accuracy(
        test_df, test_predictions, metrics_file_name)

    return fig


@capture("graph")
def plot_linear_regression(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    return plot_util(df, get_linear_regression_predictions, 'linear_metrics.csv')


@capture("graph")
def plot_ridge_regression(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    return plot_util(df, get_ridge_lasso_regression_predictions, 'ridge_metrics.csv')


@capture("graph")
def plot_lasso_regression(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    return plot_util(df, get_ridge_lasso_regression_predictions, 'lasso_metrics.csv')


@capture("graph")
def plot_combo_regression(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    return plot_util(df, get_ridge_lasso_combo_predictions, 'combined_metrics.csv')


@capture("graph")
def plot_metrics(data_frame: pd.DataFrame = None):
    linear_metrics = pd.read_csv('linear_metrics.csv', index_col='Label')
    ridge_metrics = pd.read_csv('ridge_metrics.csv', index_col='Label')
    lasso_metrics = pd.read_csv('lasso_metrics.csv', index_col='Label')
    combined_metrics = pd.read_csv('combined_metrics.csv', index_col='Label')

    color_accuracy_data = {
        'Linear': linear_metrics.loc['Color Accuracy', 'Color Accuracy'],
        'Ridge': ridge_metrics.loc['Color Accuracy', 'Color Accuracy'],
        'Lasso': lasso_metrics.loc['Color Accuracy', 'Color Accuracy'],
        'Combined': combined_metrics.loc['Color Accuracy', 'Color Accuracy'],
    }

    volatility_score_data = {
        'Linear': linear_metrics.loc['Volatility Score', 'Volatility Score'],
        'Ridge': ridge_metrics.loc['Volatility Score', 'Volatility Score'],
        'Lasso': lasso_metrics.loc['Volatility Score', 'Volatility Score'],
        'Combined': combined_metrics.loc['Volatility Score', 'Volatility Score'],
    }

    labels = ['Open', 'High', 'Low', 'Close']
    metrics = ['MAE', 'RMSE', 'R2', 'Explained Variance']
    models = ['Linear', 'Ridge', 'Lasso', 'Combined']

    data = {}
    for metric in metrics:
        data[metric] = np.array([
            [linear_metrics.loc[label, metric] for label in labels],
            [ridge_metrics.loc[label, metric] for label in labels],
            [lasso_metrics.loc[label, metric] for label in labels],
            [combined_metrics.loc[label, metric] for label in labels],
        ])

    model_colors = {
        'Linear': '#1f77b4',  # blue
        'Ridge': '#ff7f0e',   # orange
        'Lasso': '#2ca02c',   # green
        'Combined': '#d62728',  # red
    }

    ordered_metrics = ['MAE', 'RMSE', 'R2',
                       'Explained Variance', 'Color Accuracy', 'Volatility Score']
    subplot_titles = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)',
                      'R-Squared (R2)', 'Explained Variance', 'Predicted Candlestick Color Accuracy', 'Predicted Volatility Score']

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,  # Adjust spacing to prevent overlap
        vertical_spacing=0.1  # Adjust spacing between rows
    )

    for i, metric in enumerate(ordered_metrics):
        if metric in ['Color Accuracy', 'Volatility Score']:
            # Plot 'Color Accuracy' and 'Volatility Score' in the second row
            fig.add_trace(
                go.Bar(
                    x=list(color_accuracy_data.keys()) if metric == 'Color Accuracy' else list(
                        volatility_score_data.keys()),
                    y=list(color_accuracy_data.values()) if metric == 'Color Accuracy' else list(
                        volatility_score_data.values()),
                    marker_color=list(model_colors.values()),
                    showlegend=False
                ),
                row=2,
                col=2 if metric == 'Color Accuracy' else 3
            )
        else:
            for j, model in enumerate(models):
                # Plot other metrics in the first row and first two slots of the second row
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=data[metric][j],
                        name=model,
                        legendgroup=model,
                        marker_color=model_colors[model],
                        showlegend=(i == 0),
                    ),
                    row=(i // 3) + 1,
                    col=(i % 3) + 1
                )

    fig.update_layout(
        height=650,
        width=850,
        legend=dict(orientation="h", yanchor="bottom",
                    y=-0.2, xanchor="center", x=0.5),
        yaxis5=dict(range=[0, 1])
    )
    return fig


@capture("graph")
def plot_linear_regression_with_features(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    df = add_features(df)
    return plot_util(df, get_linear_regression_predictions, 'linear_metrics.csv')


@capture("graph")
def plot_ridge_regression_with_features(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    df = add_features(df)
    return plot_util(df, get_ridge_lasso_regression_predictions, 'ridge_metrics.csv')


@capture("graph")
def plot_lasso_regression_with_features(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    df = add_features(df)
    return plot_util(df, get_ridge_lasso_regression_predictions, 'lasso_metrics.csv')


@capture("graph")
def plot_combo_regression_with_features(data_frame: pd.DataFrame = None):
    df = data_frame.copy()
    df = add_features(df)
    return plot_util(df, get_ridge_lasso_combo_predictions, 'combined_metrics.csv')

# ---------------------------------------------------------------------------------#
