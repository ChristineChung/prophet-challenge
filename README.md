# prophet-challenge
module 8 challenge

My study group and I worked together during multiple sessions between Wednesday and the weekend to complete this homework assignment. There were about five of us working through the solutions together. We worked together on the entire assignment. We went step by step through each problem and referenced the teachers notes from class to talk through the code solutions. We took turns sharing code ideas to solve the questions and troubleshooted errors together. We used any info anyone learned from the TA's and BCS tutors and shared it with the group to work through all the solutions. 

# Forecasting Net Prophet

You’re a growth analyst at [MercadoLibre](http://investor.mercadolibre.com/about-us). With over 200 million users, MercadoLibre is the most popular e-commerce site in Latin America. You've been tasked with analyzing the company's financial and user data in clever ways to make the company grow. So, you want to find out if the ability to predict search traffic can translate into the ability to successfully trade the stock.

The instructions for this challenge are divided into four steps:

- **Step 1:** Find unusual patterns in hourly Google search traffic
- **Step 2:** Mine the search traffic data for seasonality
- **Step 3:** Relate the search traffic to stock price patterns
- **Step 4:** Create a time series model with Prophet

---

## Setup and Library Installation
To get started, install the necessary libraries and import them into your environment.

```bash
# Install Prophet library
!pip install prophet
python
Copy code
# Import the required libraries and dependencies
import pandas as pd
from prophet import Prophet
import datetime as dt
import numpy as np
%matplotlib inline
```

## Step 1: Find Unusual Patterns in Hourly Google Search Traffic
The data science manager asks if the Google search traffic for the company links to any financial events at the company, or if it presents random noise. To investigate, perform the following:

## Step 1a: Read the search data into a DataFrame and focus on May 2020.

```python
# Store the data in a Pandas DataFrame and set the "Date" column as the Datetime Index
df_mercado_trends = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/google_hourly_search_trends.csv",
    index_col='Date',
    parse_dates=True
).dropna()

# Slice the DataFrame to May 2020
df_may_2020 = df_mercado_trends.loc['2020/05/01':'2020/05/31']

# Plot the data to visualize May 2020
df_may_2020.plot()
Step 1b: Calculate the total search traffic for May 2020 and compare it to the monthly median.
python
Copy code
# Calculate the total search traffic for May 2020
traffic_may_2020 = df_may_2020.sum()

# Calculate the monthly median search traffic across all months
median_monthly_traffic = df_mercado_trends.groupby(
    [df_mercado_trends.index.year, df_mercado_trends.index.month]
).sum().median()

# Compare the search traffic for May 2020 with the overall monthly median value
traffic_may_2020 / median_monthly_traffic
```

Analysis
Question: Did Google search traffic increase during May 2020, when MercadoLibre released its financial results?

Answer: Yes, search traffic increased by 8.5% compared to the monthly median, with spikes around May 5th and 6th, which coincide with the firm's quarterly financial results. This suggests that the higher traffic was due to increased investor interest.

## Step 2: Mine the Search Traffic Data for Seasonality
This step explores predictable seasonal patterns in the search traffic data to guide marketing efforts. Complete the following:

## Step 2a: Group the hourly search data to plot the average traffic by the hour of the day.

```python
# Group the hourly search data by the hour of the day
df_mercado_trends.groupby(df_mercado_trends.index.hour).mean().plot()
Step 2b: Group the hourly search data to plot the average traffic by the day of the week.
python
Copy code
# Group the hourly search data by the day of the week
df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().day).mean().plot()
Step 2c: Group the hourly search data to plot the average traffic by the week of the year.
python
Copy code
# Group the hourly search data by the week of the year
df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().week).mean().plot()
```

Analysis
Question: Are there any time-based trends in the search traffic data?

Answer:

The traffic peaks in the evenings (10 PM to midnight) and starts to taper off after Monday, with the peak day being Tuesday.
The seasonality trends show a dip in search traffic during the first week of the year and in mid-October, possibly due to holidays.
Step 3: Relate the Search Traffic to Stock Price Patterns
In this step, determine if a relationship exists between the search data and the stock price. Complete the following:

## Step 3a: Read in the stock price data and concatenate it with the search traffic data.

```python
# Read in the stock price data
df_mercado_stock = pd.read_csv(
    "https://static.bc-edx.com/ai/ail-v-1-0/m8/lms/datasets/mercado_stock_price.csv",
    index_col='date',
    parse_dates=True
).dropna()

# Concatenate the stock price and search traffic data
mercado_stock_trends_df = pd.concat([df_mercado_stock, df_mercado_trends], axis=1).dropna()
```

## Step 3b: Slice the data to the first half of 2020 and plot the data.

```python
# Slice to just the first half of 2020
first_half_2020 = mercado_stock_trends_df['2020-01':'2020-06']

# Plot the close and search trends data on separate axes
first_half_2020.plot(subplots=True)
```

## Step 3c: Create new columns in the DataFrame: "Lagged Search Trends", "Stock Volatility", and "Hourly Stock Return".

```python
# Create the "Lagged Search Trends" column
mercado_stock_trends_df['Lagged Search Trends'] = df_mercado_trends['Search Trends'].shift(1)

# Create the "Stock Volatility" column with a rolling 4-period std
mercado_stock_trends_df['Stock Volatility'] = df_mercado_trends['close'].pct_change().rolling(window=4).std()

# Create the "Hourly Stock Return" column
mercado_stock_trends_df['Hourly Stock Return'] = df_mercado_trends['close'].pct_change()

# Calculate correlation between key metrics
mercado_stock_trends_df[['Stock Volatility', 'Lagged Search Trends', 'Hourly Stock Return']].corr()
```

Analysis
Question: Does a predictable relationship exist between lagged search traffic and stock price patterns?

Answer: There's a slight negative correlation between lagged search traffic and stock volatility, suggesting that increased search traffic might lead to lower stock volatility. However, there's a weak positive correlation between lagged search traffic and hourly stock return. Overall, these correlations are weak, indicating that search traffic has a limited predictive capacity.

## Step 4: Create a Time Series Model with Prophet
This step involves creating a time series model with Prophet to analyze and forecast patterns in hourly search data. Complete the following:

## Step 4a: Set up the Google search data for a Prophet forecasting model.

```python
# Prepare the data for Prophet
mercado_prophet_df = df_mercado_trends.reset_index()
mercado_prophet_df.columns = ['ds', 'y']
mercado_prophet_df.dropna()

# Fit the Prophet model
model_mercado_trends = Prophet()
model_mercado_trends.fit(mercado_prophet_df)

# Create a future dataframe for predictions (2000 hours)
future_mercado_trends = model_mercado_trends.make_future_dataframe(periods=2000, freq='H')

# Generate the forecast
forecast_mercado_trends = model_mercado_trends.predict(future_mercado_trends)
```

## Step 4b: After estimating the model, plot the forecast.

```python
# Plot the Prophet forecast
model_mercado_trends.plot(forecast_mercado_trends)
```

## Step 4c: Plot the individual time series components to understand key patterns.

```python
# Plot the components to visualize trends
model_mercado_trends.plot_components(forecast_mercado_trends)
```
Analysis
Question: What insights can be derived from the forecast and time series components?

Answer:

The near-term forecast indicates a slight decline in search traffic through the rest of 2020.
The peak traffic times are generally in the late evening (10 PM to midnight).
Tuesday is typically the busiest day of the week.
The lowest point for search traffic in the calendar year is in the last week of the year and mid-October, possibly due to holidays like Día de La Raza.