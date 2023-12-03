import pandas as pd
import os
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, ZivotAndrews
from arch.unitroot.cointegration import engle_granger, phillips_ouliaris, DynamicOLS, FullyModifiedOLS
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_coint_rank


# Get and set the current working directory
working_directory = os.path.dirname(__file__)
os.chdir(working_directory)

#import and basic cleaning
file_path=os.path.join(os.getcwd(), 'MVE_Assignment_DataSet.xlsx')
columns = ['Unnamed: 0',
           #'EN.ATM.METH.AG.KT.CE',
           #'EN.ATM.NOXE.AG.KT.CE',
           'AG.PRD.CROP.XD',
           'EN.ATM.CO2E.KT',
           'mean+tmp',
           'mean_pre',
           #'SP.POP.TOTL',
           #'NV.AGR.TOTL.KD'
           ]

data = pd.read_excel(file_path, header=1, usecols=columns)

data.rename(columns={'Unnamed: 0': 'year',
                     #'EN.ATM.METH.AG.KT.CE' : 'methane' ,
                     #'EN.ATM.NOXE.AG.KT.CE' :  'nox' ,
                     'AG.PRD.CROP.XD' :  'crop_production' ,
                     'EN.ATM.CO2E.KT' :  'co2',
                     'mean+tmp' :  'mean_temp',
                     'mean_pre': 'mean_rainfall',
                     #'SP.POP.TOTL' : 'population',
                     #'NV.AGR.TOTL.KD' :  'agricultural_GDP'     
                     }, inplace=True)
data.set_index('year', inplace=True)

data=data[['crop_production', 'mean_temp','mean_rainfall','co2']]

for col in data.columns:
    fig, axs = plt.subplots(3, 1, figsize=(12, 15)) 
    axs[0].plot(data.index, data[col], label=col)
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Level')
    axs[0].legend()
  
    axs[1].plot(data.index, np.log(data[col]), label=col)
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Log(Level)')
    axs[1].legend()

    axs[2].plot(data.index, data[col].diff(), label=col)
    axs[2].set_xlabel('Year')
    axs[2].set_ylabel('First Difference')
    axs[2].legend()

    output_directory = 'graphs'
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, f'{col}.png')
    plt.savefig(output_file_path)

    plt.show()

# Take the natural logarithm of CO2 to stabilize variance and distribution
data[['co2']] = np.log(data[['co2']])


#Part 3
# Check for serial correlation of residuals
for column_name in data.columns:
    print(f'The results of {column_name}:')
    timeseries = np.asarray(data[column_name])
    y = timeseries[1:]
    y_lag = timeseries[:-1]
    model = sm.OLS(y, y_lag)
    results = model.fit()
    
    plot_acf(results.resid, lags=10,title=f'ACF - {column_name} (Residuals)')
    output_directory = 'graphs'
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, f'{column_name}_residuals_acf.png')
    plt.savefig(output_file_path)

    # Calculate and print Durbin-Watson statistic
    dw_statistic = durbin_watson(results.resid)
    print(f"\nDurbin-Watson Statistic: {dw_statistic}")

    # Interpret Durbin-Watson statistic
    if dw_statistic < 1.5:
        print("Positive autocorrelation may be present.")
    elif dw_statistic > 2.5:
        print("Negative autocorrelation may be present.")
    else:
        print("No significant autocorrelation detected.")
    print()
    print()


### Testing 

# Define function to apply various tests for a unit root in time series: Dickey-Fuller, Augmented Dickey-Fuller, Phillips Perron, and KPSS. Also correcting for possible deterministic components in the data: 'n' (no deterministic components), 'c' (a constant), 'ct' (a constant and trend.)
def unit_root_test(data: pd.DataFrame, diff: int):
    
    df_results = pd.DataFrame(columns=['Variable', 'Differenced', 'Deterministic Component', 'DF', 'ADF', 'PP', 'KPSS', 'ZA'])
    
    differences = 0
    if diff == 0:
        data = data
    else:
        for i in range(diff):
            data = data.diff().dropna()
            differences += 1
      
    for col in data.columns:
    # 1. c
        df_df= adfuller(data[col],autolag=None, maxlag=0, regression='c')
        df_adf = adfuller(data[col], autolag=None, maxlag=1, regression='c')
        p_value_pp = PhillipsPerron(data[col], trend="c").pvalue
        df_kpss = kpss(data[col], nlags=1, regression='c')
        df_za = ZivotAndrews(data[col], trend='c', lags=1).pvalue
        df_results = pd.concat([df_results, pd.DataFrame({'Variable': [col],
                                                          'Differenced': differences,
                                                          'Deterministic Component': ['c'],
                                                          'DF': [df_df[1]],
                                                          'ADF': [df_adf[1]],
                                                          'PP': [p_value_pp],
                                                          'KPSS': [df_kpss[1]],
                                                          'ZA': [df_za]
                                                           })], ignore_index=True)
    
        # 2. ct
        df_df= adfuller(data[col],autolag=None,maxlag=0, regression='ct')
        df_adf = adfuller(data[col], autolag=None, maxlag=1, regression='ct')
        p_value_pp = PhillipsPerron(data[col], trend="ct").pvalue
        df_kpss = kpss(data[col], nlags=1, regression='ct')
        df_za = ZivotAndrews(data[col], trend='ct', lags=1).pvalue
        df_results = pd.concat([df_results, pd.DataFrame({'Variable': [col],
                                                          'Differenced': differences,
                                                          'Deterministic Component': ['ct'],
                                                          'DF': [df_df[1]],
                                                          'ADF': [df_adf[1]],
                                                          'PP': [p_value_pp],
                                                          'KPSS': [df_kpss[1]],
                                                          'ZA': [df_za]
                                                           })], ignore_index=True)
    
        # 3. n
        df_df= adfuller(data[col],autolag=None,maxlag=0, regression='n')
        df_adf = adfuller(data[col], autolag=None, maxlag=1, regression='n')
        p_value_pp = PhillipsPerron(data[col], trend="n").pvalue
        df_kpss = kpss(data[col], nlags=1)
        df_za = ZivotAndrews(data[col], lags=1).pvalue
        df_results = pd.concat([df_results, pd.DataFrame({'Variable': [col],
                                                          'Differenced': differences,
                                                          'Deterministic Component': ['n'],
                                                          'DF': [df_df[1]],
                                                          'ADF': [df_adf[1]],
                                                          'PP': [p_value_pp],
                                                          'KPSS': [df_kpss[1]],
                                                          'ZA': [df_za]
                                                           })], ignore_index=True)

    return df_results


# Test for unit roots in level data, 1st differences, and 2nd differences
for i in range(3):
    global_variable_name = f"df_result_diff{i}"
    globals()[global_variable_name] = unit_root_test(data, i).round(4)
    print(globals()[global_variable_name])


# Part 4 Robert: Test for (no)-cointegration using some of the techniques discussed during the course

# All possible cointegration relationships
coint_relations = {
                   'test 1':{'y': 'crop_production', 'x': ['mean_temp', 'mean_rainfall', 'co2']},
                   'test 2':{'y': 'crop_production', 'x': ['mean_temp', 'mean_rainfall']},
                   'test 3':{'y': 'crop_production', 'x': ['mean_rainfall','co2']},
                   'test 4':{'y': 'crop_production', 'x': ['mean_temp', 'co2']},
                   'test 5':{'y': 'crop_production', 'x': ['mean_temp']},
                   'test 6':{'y': 'crop_production', 'x': ['mean_rainfall']},
                   'test 7':{'y': 'crop_production', 'x': ['co2']}
                   }
                   

# Defining function to test cointegration according to Engle-Granger, Phillips Ouliaris, and Johansen
def cointegration_test(data:pd.DataFrame, y:str, x:list):
    eg_coint_result = engle_granger(data[y], data[x])
    po_coint_result = phillips_ouliaris(data[y], data[x], test_type="Zt")
    print(f"Dependent variable: {y}")
    print(f"Independent variable(s): {x}")
    print()
    
    # Engle-Granger cointegration
    print(f"The p-value of the Engle-Granger cointegration test is: {eg_coint_result.pvalue}")
    if eg_coint_result.pvalue < 0.05:
        print("The variables are cointegrated")
    else:
        print("The variables are not cointegrated")
    print()
    
    # Phillips Ouliaris cointegration
    print(f"The p-value of the Phillips Ouliaris cointegration test is: {po_coint_result.pvalue}")
    if eg_coint_result.pvalue < 0.05:
        print("The variables are cointegrated")
    else:       
        print("The variables are not cointegrated")
    print()
    
    # Johansen Trace
    johansen_df = pd.concat([data[y], data[x]], axis=1)
    trace_coint_result = select_coint_rank(endog=johansen_df, det_order=0, k_ar_diff=1, method='trace')
    maxeig_coint_result = select_coint_rank(endog=johansen_df, det_order=0, k_ar_diff=1, method='maxeig')
    print(f"There are {trace_coint_result.rank} cointegrating vectors according to the Johansen Trace test")
    print(f"There are {maxeig_coint_result.rank} cointegrating vectors according to the Johansen Maximum Eigenvalue test")
    print()

# Testing for cointegration
for i in coint_relations:
    cointegration_test(data, coint_relations[i]['y'], coint_relations[i]['x'])
    print()


# Define function for cointegrating regressions
def cointegration_regression(data: pd.DataFrame, y:str, x:list):
    static_ols = sm.OLS(data[y], sm.add_constant(data[x])).fit()
    dynamic_ols = DynamicOLS(data[y], data[x], lags=1).fit()
    fully_mod_ols = FullyModifiedOLS(data[y], data[x]).fit()

    return static_ols.summary(), dynamic_ols.summary(), fully_mod_ols.summary()

# Estimate cointegrating regressions
for i in coint_relations:
    for j in range(3):
        print(cointegration_regression(data, coint_relations[i]['y'], coint_relations[i]['x'])[j])





#Part 4
# Assuming the data is I(1), then in the first part we are performing Engle-Granger, ADF and Philip-Ouliaris tests:

results = []

# Engle-Granger Cointegration Test
for col1 in data.columns:
    for col2 in data.columns:
        if col1 != col2:
            eg_test_stat, eg_p_value, _ = coint(data[col1], data[col2], trend='c')
            results.append((f"Engle-Granger Test ({col1}, {col2})", eg_test_stat, eg_p_value))

# Johansen Cointegration Test
johansen_results = coint_johansen(data.values, det_order=1, k_ar_diff=1)
trace_statistic = johansen_results.lr1
max_eig_statistic = johansen_results.lr2
trace_critical_values = johansen_results.cvm[:, 1]
max_eig_critical_values = johansen_results.cvm[:, 0]
results.append(("Johansen Trace Test", trace_statistic, "N/A"))
results.append(("Johansen Max Eigenvalue Test", max_eig_statistic, "N/A"))

# Phillips-Ouliaris Cointegration Test 
for col1 in data.columns:
    for col2 in data.columns:
        if col1 != col2:
            diff_series = data[col1] - data[col2]
            lags = 1  # You can adjust the number of lags as needed
            lagged_diff = diff_series.shift(1).dropna()
            lagged_levels = data[col1].shift(1).dropna(), data[col2].shift(1).dropna()

            # Regression
            model = PhillipsPerron(lagged_diff, trend='c', lags=lags)
            po_test_stat = model.stat
            po_p_value = model.pvalue

            results.append((f"Phillips-Ouliaris Test ({col1}, {col2})", po_test_stat, po_p_value))

# ADF Cointegration Test
for col1 in data.columns:
    for col2 in data.columns:
        if col1 != col2:
            adf_test_result = adfuller(data[col1] - data[col2], autolag='AIC')
            adf_test_stat = adf_test_result[0]
            adf_p_value = adf_test_result[1]
            critical_values = adf_test_result[4]
            results.append((f"ADF Test ({col1}, {col2})", adf_test_stat, adf_p_value, critical_values))

# Results
columns = ["Test", "Test Statistic", "P-Value", "Critical Values"]
df_results = pd.DataFrame(results, columns=columns)

print(tabulate(df_results, headers='keys', tablefmt='fancy_grid'))

# In the second part, we perform Maximum Likelihood based tests based on Johansen Trace and Maximum Eigenvalue tests:
results = []

# Johansen Cointegration Test
johansen_results = coint_johansen(data.values, det_order=1, k_ar_diff=1)

# Trace Statistic and Critical Values
trace_statistic = johansen_results.lr1
trace_critical_values = johansen_results.cvt[:, 2]  # Adjust the index based on your number of variables

# Maximum Eigenvalue Statistic and Critical Values
max_eig_statistic = johansen_results.lr2
max_eig_critical_values = johansen_results.cvm[:, 2]  # Adjust the index based on your number of variables

# Eigenvalues
eigenvalues = johansen_results.eig

results.append(("Johansen Trace Test", trace_statistic, trace_critical_values))
results.append(("Johansen Max Eigenvalue Test", max_eig_statistic, max_eig_critical_values))

# Results
columns = ["Test", "Test Statistic", "Critical Values"]
df_results = pd.DataFrame(results, columns=columns)

print(tabulate(df_results, headers='keys', tablefmt='fancy_grid'))
