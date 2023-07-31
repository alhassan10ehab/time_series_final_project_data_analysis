
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

class EDA:
    def __init__(self , df):
        self.df = df
 
    #------------------------------------------------------------------------------------
    '''                             Info ^_^                                      '''
    #----------------------------------------------------------------------------------
    def summary(self):
        print(self.df.info())
        return self.df.describe(include = 'all')
    #------------------------------------------------------------------------------------
    '''                             Check null ^_^                                      '''
    #----------------------------------------------------------------------------------
    '''
    This function counts null in each column in the dataframe and calculate the percent of nulls in the column then return the 
    dataframe consist of 2 columns :  one contains count of null values in each column and second contains percent 
    '''
    
    def null_values(self , plot =True , count_zero_values = True):
        null_val = pd.DataFrame(self.df.isnull().sum())
        null_val.columns = ['null_val']
        null_val['percent_null'] = round(null_val['null_val'] / len(self.df.index), 5) * 100
        null_val = null_val.sort_values('null_val', ascending = False)

        if plot:
            ax = sns.heatmap(self.df.isnull(), cbar=False)
            msno.heatmap(self.df)
            msno.dendrogram(self.df)
            msno.bar(self.df)
            plt.show()

        if count_zero_values:
            null_val['zero_value'] = self.df.apply(self.count_zeros)
            null_val['total_percent'] = round((null_val['null_val'] + null_val['zero_value']) / len(self.df.index), 5) * 100

        return null_val
    
    
    def count_zeros(self, column):
        # Get the count of Zeros in column 
        count = (column == 0).sum()
        return count   
    

    #------------------------------------------------------------------------------------
    '''                             Check duplication ^_^                                  '''
    #----------------------------------------------------------------------------------
    '''
    This function counts duplicated rows in the dataframe 
    '''
    def duplicated_values(self):
        return print("Number of duplicated rows" , self.df.duplicated().sum())
    
   

    #------------------------------------------------------------------------------------
    '''                             CHECK CONSTANT FEATURES ^_^                               '''
    #----------------------------------------------------------------------------------
    '''
    This function returns the columns that contain one value a cross all samples
    '''
    def constant_columns(self):
        constant_columns = [col  for col in self.df.columns if (self.df[col].nunique()) == 1]
        return constant_columns
    
    #------------------------------------------------------------------------------------
    '''                             CHECK cardinality FEATURES ^_^                               '''
    #----------------------------------------------------------------------------------
    '''
    calculate unique values in each column and returns dataframe consists of count and percent. This helps us to find column that have 
    high cardinality 
    '''
    def cardinality(self):
        unique_val = pd.DataFrame(np.array([len(self.df[col].unique()) for col in self.df.columns ]) , index=self.df.columns)
        unique_val.columns = ['unique_val']
        unique_val['percent_'] = round(unique_val['unique_val'] / len(self.df.index), 2) * 100
        unique_val = unique_val.sort_values('percent_', ascending = False)
        return unique_val
    


    #------------------------------------------------------------------------------------
    '''                             Check the redundant_features ^_^                               '''
    #----------------------------------------------------------------------------------   
  
    '''
    This Function check if there is a high correlation between 2 features .we set a thershold to 0.98. if any 2 features 
    have a correlation larger than 0.95, put them in list .then return correlation matrix and list 
    
    
    '''
    def redundant_features(self):
        #Creating the Correlation matrix
        cor_matrix = self.df.corr().abs()
        #Select the upper triangular
        upper_tri =  pd.DataFrame(cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool)))
        #Select the columns which are having absolute correlation greater than 0.98 and making a list of those columns 
        self.features_high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if len(self.features_high_corr) == 0 :
            print("There is no redundant features")
            print("*" * 50)
            self.features_high_corr = "empty"
            return  upper_tri 
        else:
            print(self.features_high_corr)
            return  upper_tri
    #------------------------------------------------------------------------------------
    '''                             drop columns  ^_^                                  '''
    #----------------------------------------------------------------------------------
    '''
    This function drop columns and rows that contain null values.
    '''
    
    def drop_col(self , cols , options = False):

        if options :
            cols_to_drop = list(set(self.features_high_corr)|set(cols))
        else : 
            cols_to_drop = cols
        # Drop columns
        self.df.drop(cols_to_drop , axis = 1 , inplace = True ) 
        #Drop rows with null values
        self.df.dropna(axis = 0 , inplace = True)
        return self.df
    
    #------------------------------------------------------------------------------------
    '''                           Plot features vs time ^_^                          '''
    #----------------------------------------------------------------------------------
    '''
    This is a method that visualizes the features of a dataset plotted against the time axis ,which can be helpful for identifying trends, patterns, 
    and anomalies in the data.
    '''
    def visualize_features_vs_Date(self):
        plt.figure(figsize = (15,15))
        colors = ['#FF7F50', '#DDA0DD', '#66CDAA', '#BC8F8F']
        for indx, col in enumerate(self.df.columns):
            plt.subplot(len(self.df.columns) + 1, 1, indx + 1)
            plt.plot(self.df[col], color= np.random.choice(colors));
            plt.ylabel(col, fontsize=16)
            plt.grid()
        plt.xlabel('Date', fontsize=16)
        
        
    #---------------------------------------------------------------------------------------
    '''                           Check Stationery ^_^                                 '''
    #--------------------------------------------------------------------------------------
    '''
    This function includes three different methods to check stationery:
    
    - Autocorelation plots show how correlated are values at time t with the next values in time t+1,t+2,..t+n. 
    If the data would be non-stationary the autocorrelation values will be highly correlated with distant points in time showing possible seasonalities or trends.
    Stationary series autocorrelation values will quickly decrease over time t. This shows us that no information is carried over time and then the series should be constant over time.
    
    - Rolling mean and standard deviation should be constant over time in order to have a stationary time series
    
    - The Augmented Dickey-Fuller test is a type of statistical test 
        - Null Hypothesis (H0): If failed to be rejected, it suggests the time series is non-stationary. 
        - Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series is stationary. 
    '''
    def check_stationery(self , lags_ , col , window_size):
    
        #---------------------- Autocorrelation and Partial autocorrelation plots ^_^ -----------------------
        plt.figure(figsize = (10,5))
        ax = plt.subplot(1, 2,1)
        plot_acf(self.df[col], lags=lags_ , ax = ax)
        ax = plt.subplot(1,2,2)
        plot_pacf(self.df[col], lags=lags_ , ax = ax)
        
        #------------------------- Rolling mean and standard deviation ^_^ -------------------------------
        rol_mean =  self.df[col].rolling(window = window_size , center = True , min_periods = int(window_size/2) ).mean()
        rol_std =  self.df[col].rolling(window = window_size , center = True , min_periods = int(window_size/ 2) ).std()
        
        plt.figure()
        plt.plot(self.df[col] , label = 'Orignal data')
        plt.plot(rol_mean, label = 'Rolling Mean' )
        plt.plot(rol_std , label = 'Rolling Std')
        plt.legend()
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        
        #--------------------------- Augmented Dickey-Fuller test ^_^ -------------------------------------------
        
        p_value = adfuller(self.df[col])[1]
        hypothesis_result = print("Reject Null Hypothesis and The time series is  stationery") if p_value <= 0.05 else print("We cann't reject null hypothesis and The time series is non stationery")
        hypothesis_result
    
    def Check_Noise(self):
        fig = plt.figure(figsize=(12, 7))
        layout = (2, 2)
        hist_ax = plt.subplot2grid(layout, (0, 0))
        ac_ax = plt.subplot2grid(layout, (1, 0))
        hist_std_ax = plt.subplot2grid(layout, (0, 1))
        mean_ax = plt.subplot2grid(layout, (1, 1))

        air_pollution.pollution_today.hist(ax=hist_ax)
        hist_ax.set_title("Original series histogram")

        plot_acf(series, lags=30, ax=ac_ax)
        ac_ax.set_title("Autocorrelation")

        mm = air_pollution.pollution_today.rolling(7).std()
        mm.hist(ax=hist_std_ax)
        hist_std_ax.set_title("Standard deviation histogram")

        mm = air_pollution.pollution_today.rolling(30).mean()
        mm.plot(ax=mean_ax)
        mean_ax.set_title("Mean over time")
        
        