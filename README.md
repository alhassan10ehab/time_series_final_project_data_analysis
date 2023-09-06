# Problem statement
Our problem is to investigate the effectiveness of Transformer based models for time series forecasting and compare their performance with LSTM and linear models named LTSF Linear, such as Linear.
the patchTST paper: https://arxiv.org/abs/2211.14730
linear models and fedformer paper: https://arxiv.org/abs/2205.13504

# Data
There are five datasets were used four of them were electrical power consumption time series from collected from industrial sites. Each data point consists of 8 features, including the date of the point, the predictive value ”oil temperature”, and 6 different types of external power load features. The features are “date”, ”HUFL”, ”HULL”, ”MUFL”, ”MULL”, ”LUFL”, ”LULL” and ”OT”.

The another dataset (Hourly energy demand generation and weather) contains 4 years of electrical consumption, generation, pricing, and weather data for Spain. Consumption and generation data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España.

# models
**Patch-TST**

**Linear-LSTF**

**FEDformer**

**LSTM**

# results
![Screenshot (40)](https://github.com/alhassan10ehab/time_series_final_project_data_analysis/assets/130251324/77308c61-c682-4879-ba52-c37fd732d77b)
