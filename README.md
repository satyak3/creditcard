# Introduction

In this kernel, we are going to predict whether a credit card is fraud or not using Machine Learning.

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

Due to confidentiality issues, the input variables are transformed into numerical using PCA transformations.

The dataset is taken from kaggle <a href='https://www.kaggle.com/mlg-ulb/creditcardfraud' target='_blank'>here</a>.

# Importing the required Python libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

%matplotlib inline
sns.set()
warnings.simplefilter('ignore')
```

# Data Preprocessing

Let's get the dataset into a pandas dataframe.


```python
data = pd.read_csv('creditcard.csv')
df = data.copy() # To keep the data as backup
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
df.shape
```




    (284807, 31)




```python
df.isnull().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
df.dtypes
```




    Time      float64
    V1        float64
    V2        float64
    V3        float64
    V4        float64
    V5        float64
    V6        float64
    V7        float64
    V8        float64
    V9        float64
    V10       float64
    V11       float64
    V12       float64
    V13       float64
    V14       float64
    V15       float64
    V16       float64
    V17       float64
    V18       float64
    V19       float64
    V20       float64
    V21       float64
    V22       float64
    V23       float64
    V24       float64
    V25       float64
    V26       float64
    V27       float64
    V28       float64
    Amount    float64
    Class       int64
    dtype: object




```python
df.Time.tail(15)
```




    284792    172774.0
    284793    172775.0
    284794    172777.0
    284795    172778.0
    284796    172780.0
    284797    172782.0
    284798    172782.0
    284799    172783.0
    284800    172784.0
    284801    172785.0
    284802    172786.0
    284803    172787.0
    284804    172788.0
    284805    172788.0
    284806    172792.0
    Name: Time, dtype: float64




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>3.919560e-15</td>
      <td>5.688174e-16</td>
      <td>-8.769071e-15</td>
      <td>2.782312e-15</td>
      <td>-1.552563e-15</td>
      <td>2.010663e-15</td>
      <td>-1.694249e-15</td>
      <td>-1.927028e-16</td>
      <td>-3.137024e-15</td>
      <td>...</td>
      <td>1.537294e-16</td>
      <td>7.959909e-16</td>
      <td>5.367590e-16</td>
      <td>4.458112e-15</td>
      <td>1.453003e-15</td>
      <td>1.699104e-15</td>
      <td>-3.660161e-16</td>
      <td>-1.206049e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>



## Checking the frequency of frauds before moving forward


```python
df.Class.value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64




```python
sns.countplot(x=df.Class, hue=df.Class)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ff9d53ebe0>




![png] (/graph/output_13_1.png)


By looking at the above statistics, we can see that the data is highly imbalanced. Only 492 out of 284807 are fraud.

## Checking the distribution of amount


```python
plt.figure(figsize=(10, 5))
sns.distplot(df.Amount)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ff9d536160>




![png](/graph/output_16_1.png)


Since, it is a little difficult to see. Let's engineer a new feature of bins.


```python
df['Amount-Bins'] = ''
```

Now, let's set the bins and their labels.


```python
def make_bins(predictor, size=50):
    '''
    Takes the predictor (a series or a dataframe of single predictor) and size of bins
    Returns bins and bin labels
    '''
    bins = np.linspace(predictor.min(), predictor.max(), num=size)

    bin_labels = []

    # Index of the final element in bins list
    bins_last_index = bins.shape[0] - 1

    for id, val in enumerate(bins):
        if id == bins_last_index:
            continue
        val_to_put = str(int(bins[id])) + ' to ' + str(int(bins[id + 1]))
        bin_labels.append(val_to_put)
    
    return bins, bin_labels
```


```python
bins, bin_labels = make_bins(df.Amount, size=10)
```

Now, adding bins in the column Amount-Bins.


```python
df['Amount-Bins'] = pd.cut(df.Amount, bins=bins,
                           labels=bin_labels, include_lowest=True)
df['Amount-Bins'].head().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amount-Bins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0 to 2854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0 to 2854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0 to 2854</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0 to 2854</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0 to 2854</td>
    </tr>
  </tbody>
</table>
</div>



Let's plot the bins.


```python
df['Amount-Bins'].value_counts()
```




    0 to 2854         284484
    2854 to 5709         285
    5709 to 8563          28
    8563 to 11418          4
    11418 to 14272         3
    17127 to 19982         2
    22836 to 25691         1
    19982 to 22836         0
    14272 to 17127         0
    Name: Amount-Bins, dtype: int64




```python
plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df)
plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), <a list of 9 Text xticklabel objects>)




![png](/graph/output_26_1.png)


Since, count of values of Bins other than '0 to 2854' are difficult to view. Let's not insert the first one.


```python
plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df[~(df['Amount-Bins'] == '0 to 2854')])
plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), <a list of 9 Text xticklabel objects>)




![png](/graph/output_28_1.png)


We can see that mostly the amount is between 0 and 2854 euros. 

# Predictive Modelling

Let's predict whether a credit card is fraud or not using machine learning.

## One-hot encoding the Amount-Bins

Since, for classification, we need to pass the data in numerical form. That's why we need to One-Hot encode the Amount-Bins column.<br>
```Note: We can also label encode values.```


```python
df_encoded = pd.get_dummies(data=df, columns=['Amount-Bins'])
df = df_encoded.copy()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>Class</th>
      <th>Amount-Bins_0 to 2854</th>
      <th>Amount-Bins_2854 to 5709</th>
      <th>Amount-Bins_5709 to 8563</th>
      <th>Amount-Bins_8563 to 11418</th>
      <th>Amount-Bins_11418 to 14272</th>
      <th>Amount-Bins_14272 to 17127</th>
      <th>Amount-Bins_17127 to 19982</th>
      <th>Amount-Bins_19982 to 22836</th>
      <th>Amount-Bins_22836 to 25691</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



## Breaking the dataset into training and testing

First, separating the response variable from the explanatory variables.


```python
X = df.drop(labels='Class', axis=1)
Y = df['Class']

X.shape, Y.shape
```




    ((284807, 39), (284807,))




```python
from sklearn.model_selection import train_test_split
```


```python
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, random_state=42, test_size=0.3, shuffle=True)

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)
```

    (199364, 39) (199364,)
    (85443, 39) (85443,)
    

## Applying Machine Learning Algorithms

Let's apply different Machine Learning Algorithms then compare their metrics to select the most suitable ML algorithm. Algorithms to be used are:
* Logistic Regression
* Support Vector Machine
* Naive Bayes
* K-Nearest Neighbors
* Random Forest
* Ada Boost
* XGBoost

The metrics we'll use initially are:
* Accuracy
* Precision
* F1-Score

The main metrics we'll look at are (Reason is mentioned later):
* Recall
* AUC/RUC Curve

### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
```


```python
# Training the algorithm
lr_model.fit(xtrain, ytrain)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)




```python
# Predictions on training and testing data
lr_pred_train = lr_model.predict(xtrain)
lr_pred_test = lr_model.predict(xtest)
```

Before going further into metrics, let's first decide either Type-I or Type-II error is more important to consider.<br><br>
```Type-I  Error or False Positives:``` False Positives are the ones which are actually not fraud but the prediction said that they are fraud.<br>
```Type-II Error or False Negatives:``` False Negatives are the ones which are actually fraud but the system said that they aren't.

Well, we can say that Type-II Error is more significant because we don't want system to have a fraudulent credit card because that can be more dangerous.

So, for Type-II Error, We can say that **recall** is the important metric.


```python
# Importing the required metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
```

Let's first look at the **confusion matrix**.


```python
tn, fp, fn, tp = confusion_matrix(ytest, lr_pred_test).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Fraud</th>
      <th>Predicted Not Fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fraud</th>
      <td>82</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Not Fraud</th>
      <td>13</td>
      <td>85294</td>
    </tr>
  </tbody>
</table>
</div>



Let's draw a heatmap for the above confusion matrix.


```python
sns.heatmap(conf_matrix, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1ffa0a89320>




![png](/graph/output_52_1.png)


Heatmap also suggests that the data is highly imbalanced.

Let's look at the accuracy score.


```python
lr_accuracy = accuracy_score(ytest, lr_pred_test)
lr_accuracy
```




    0.9992158515033414



We can see here that accuracy is great. Around 99%.<br>
```BUT WAIT```<br>
We know that the dataset is highly unbalanced and accuracy takes into account the whole confusion matrix. So we can say that this measure is not suitable.

Let's look at precision and recall.


```python
lr_precision = precision_score(ytest, lr_pred_test)
lr_precision
```




    0.8631578947368421



Recall:


```python
lr_recall = recall_score(ytest, lr_pred_test)
lr_recall
```




    0.6029411764705882



Recall is very low in case of logistic regression. However, we may try to increase it by increasing the complexity of the model.

Although, let's check the recall for training dataset to get the idea of any overfitting we may be having.


```python
lr_recall_train = recall_score(ytrain, lr_pred_train)
lr_recall_train
```




    0.6376404494382022



Well, we can see that the delta is small, only around 0.03. So, we can say that the model is not overfitting.

Let's look at the F1-Score. F1-Score may tell us that one of the precision or recall is very low.


```python
from sklearn.metrics import f1_score
```


```python
lr_f1 = f1_score(ytest, lr_pred_test)
lr_f1
```




    0.7099567099567099



Let's look at the classification report.


```python
from sklearn.metrics import classification_report
```


```python
print(classification_report(ytest, lr_pred_test))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     85307
               1       0.86      0.60      0.71       136
    
       micro avg       1.00      1.00      1.00     85443
       macro avg       0.93      0.80      0.85     85443
    weighted avg       1.00      1.00      1.00     85443
    
    

Let's look at the ROC curve.

Now, for the ROC Curve, we need the probabilites of Fraud happening (which is the probability of occurance of 1)


```python
lr_pred_test_prob = lr_model.predict_proba(xtest)[:, 1]
```

Now, to draw the ROC Curve, we need to have ```True Positive Rate``` and ```False Positive Rate```.


```python
from sklearn.metrics import roc_curve, roc_auc_score
```


```python
fpr, tpr, threshold = roc_curve(ytest, lr_pred_test_prob)
```

Also, let's get the auc score.


```python
lr_auc = roc_auc_score(ytest, lr_pred_test_prob)
lr_auc
```




    0.9646767143445232



Now, let's define a function to plot the roc curve.


```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve', fontsize=15)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel('False Positive Rates', fontsize=15)
    plt.ylabel('True Positive Rates', fontsize=15)
    plt.legend(loc='best')
    
    plt.show()
```

Plotting ROC Curve.


```python
plot_roc_curve(fpr=fpr, tpr=tpr, label="AUC = %.3f" % lr_auc)
```


![png](/graph/output_82_0.png)      `   `


AUC is quite good. i.e. 0.965. Based on the data being highly imbalanced, we'll only check the AUC metric in later algorithms.

#### Model Complexity

Let's try to train the Logistic Regression models on the 2nd degree of polynomials. Not going further 2nd degree because features are already too much. Otherwise, computer gives the MemoryError.


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
# Getting the polynomial features
poly = PolynomialFeatures(degree=2)
xtrain_poly = poly.fit_transform(xtrain)
xtest_poly = poly.fit_transform(xtest)

# Training the model
model = LogisticRegression()
model.fit(xtrain_poly, ytrain)

# Getting the probabilities
train_prob = model.predict_proba(xtrain_poly)[:, 1]
test_prob = model.predict_proba(xtest_poly)[:, 1]

# Computing the ROC Score
roc_auc_score(ytrain, train_prob), roc_auc_score(ytest, test_prob)
```




    (0.9089129413350895, 0.9033804549519763)



Plotting ROC Curve for the Test data.


```python
fpr_poly, tpr_poly, threshold_poly = roc_curve(ytest, test_prob)
```


```python
plot_roc_curve(fpr=fpr_poly, tpr=tpr_poly, label='AUC = %.3f' %  roc_auc_score(ytest, test_prob))
```


![png](/graph/output_90_0.png)


First degree is better in Logistic Regression case which gives 0.965 AUC Score.

Let's also check the Recall in case of model complexity.


```python
recall_score(ytest, model.predict(xtest_poly))
```




    0.7426470588235294



Recall has increased when the model is made complex.

### Support Vector Machine

Let's try the Support Vector Machine algorithm.

Now, for support vector machines, we need to train the model after scaling the features. Let's first do that.


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
mms = MinMaxScaler()
```


```python
# Let's first check the head of the explanatory variables which are to be scaled.
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>Amount</th>
      <th>Amount-Bins_0 to 2854</th>
      <th>Amount-Bins_2854 to 5709</th>
      <th>Amount-Bins_5709 to 8563</th>
      <th>Amount-Bins_8563 to 11418</th>
      <th>Amount-Bins_11418 to 14272</th>
      <th>Amount-Bins_14272 to 17127</th>
      <th>Amount-Bins_17127 to 19982</th>
      <th>Amount-Bins_19982 to 22836</th>
      <th>Amount-Bins_22836 to 25691</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>149.62</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>2.69</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>378.66</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>123.50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>69.99</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
X_scaled = mms.fit_transform(X)
```


```python
X_scaled = pd.DataFrame(data=X_scaled, columns=X.columns)
X_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>Amount</th>
      <th>Amount-Bins_0 to 2854</th>
      <th>Amount-Bins_2854 to 5709</th>
      <th>Amount-Bins_5709 to 8563</th>
      <th>Amount-Bins_8563 to 11418</th>
      <th>Amount-Bins_11418 to 14272</th>
      <th>Amount-Bins_14272 to 17127</th>
      <th>Amount-Bins_17127 to 19982</th>
      <th>Amount-Bins_19982 to 22836</th>
      <th>Amount-Bins_22836 to 25691</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.935192</td>
      <td>0.766490</td>
      <td>0.881365</td>
      <td>0.313023</td>
      <td>0.763439</td>
      <td>0.267669</td>
      <td>0.266815</td>
      <td>0.786444</td>
      <td>0.475312</td>
      <td>...</td>
      <td>0.005824</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.978542</td>
      <td>0.770067</td>
      <td>0.840298</td>
      <td>0.271796</td>
      <td>0.766120</td>
      <td>0.262192</td>
      <td>0.264875</td>
      <td>0.786298</td>
      <td>0.453981</td>
      <td>...</td>
      <td>0.000105</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000006</td>
      <td>0.935217</td>
      <td>0.753118</td>
      <td>0.868141</td>
      <td>0.268766</td>
      <td>0.762329</td>
      <td>0.281122</td>
      <td>0.270177</td>
      <td>0.788042</td>
      <td>0.410603</td>
      <td>...</td>
      <td>0.014739</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000006</td>
      <td>0.941878</td>
      <td>0.765304</td>
      <td>0.868484</td>
      <td>0.213661</td>
      <td>0.765647</td>
      <td>0.275559</td>
      <td>0.266803</td>
      <td>0.789434</td>
      <td>0.414999</td>
      <td>...</td>
      <td>0.004807</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000012</td>
      <td>0.938617</td>
      <td>0.776520</td>
      <td>0.864251</td>
      <td>0.269796</td>
      <td>0.762975</td>
      <td>0.263984</td>
      <td>0.268968</td>
      <td>0.782484</td>
      <td>0.490950</td>
      <td>...</td>
      <td>0.002724</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



Now, let's train test split on the scaled data.


```python
xtrainS, xtestS, ytrainS, ytestS = train_test_split(
    X_scaled, Y, random_state=42, test_size=0.30, shuffle=True)
```


```python
print(xtrainS.shape, ytrainS.shape)
print(xtestS.shape, ytestS.shape)
```

    (199364, 39) (199364,)
    (85443, 39) (85443,)
    


```python
from sklearn.svm import SVC
```


```python
svc_model = SVC(kernel='linear', probability=True)
```


```python
svc_model.fit(xtrainS, ytrainS)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False)




```python
svc_pred = svc_model.predict(xtestS)
```

Let's first check the recall score.


```python
svc_recall = recall_score(ytestS, svc_pred)
```


```python
svc_recall
```




    0.8014705882352942



Recall quite increased in case of SVC.


```python
svc_pred_prob = svc_model.predict_proba(xtestS)[:, 1]
```

Now, let's draw the ROC Curve.


```python
# First, getting the auc score
svc_auc = roc_auc_score(ytestS, svc_pred_prob)

# Now, let's get the fpr and tpr
fpr, tpr, threshold = roc_curve(ytestS, svc_pred_prob)

# Now, let's draw the curve
plot_roc_curve(fpr, tpr, 'AUC: %.3f' % svc_auc)
```


![png](output_116_0.png)


The score AUC Score SVC gave is also pretty great. But it's still less than Logistic Regression Model. But the Recall increased significantly.

#### Tuning the Hyper-parameters

Now, let's tune some of the hyper-parameters of SVM and then compare the scores.


```python
# For Kernel = rbf
tuned_rbf = {'kernel': ['rbf'], 'gamma': [
    1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}

# For kernel = sigmoid
tuned_sigmoid = {'kernel': ['sigmoid'], 'gamma': [
    1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}

# For kernel = linear
tuned_linear = {'kernel': ['linear'], 'C': [
    0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
```


```python
from sklearn.model_selection import RandomizedSearchCV
```


```python
rs_rbf = RandomizedSearchCV(estimator=SVC(probability=True), 
        param_distributions=tuned_rbf, n_iter=500, n_jobs=4, scoring='roc_auc')

rs_sigmoid = RandomizedSearchCV(estimator=SVC(probability=True), 
        param_distributions=tuned_sigmoid, n_iter=500, n_jobs=4, scoring='roc_auc')

rs_linear = RandomizedSearchCV(estimator=SVC(probability=True), 
        param_distributions=tuned_linear, n_iter=500, n_jobs=4, scoring='roc_auc')
```

**For kernel rbf:**


```python
rs_rbf.fit(xtrainS, ytrainS)
```




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
              estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
              fit_params=None, iid='warn', n_iter=500, n_jobs=4,
              param_distributions={'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001, 1e-05], 'C': [0.001, 0.1, 0.1, 10, 25, 50, 100, 1000]},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='roc_auc', verbose=0)




```python
rs_rbf.best_estimator_
```




    SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
svc_rbf_best_est = rs_rbf.best_estimator_
```

Let's fit the model on the best rbf estimator.


```python
svc_rbf_best_est.fit(xtrainS, ytrainS)
```




    SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
svc_rbf_best_est_pred = svc_rbf_best_est.predict(xtestS)
```


```python
svc_rbf_best_est_pred_proba = svc_rbf_best_est.predict_proba(xtestS)[:, 1]
```

Getting the AUC Score.


```python
svc_rbf_auc = roc_auc_score(ytestS, svc_rbf_best_est_pred_proba)
```

Getting the Recall too.


```python
svc_rbf_recall = recall_score(ytestS, svc_rbf_best_est_pred)
svc_rbf_recall
```




    0.8308823529411765



We can see that in this model, both recall and ROC Score are great. Let's draw the ROC Curve.


```python
fpr, tpr, threshold = roc_curve(ytestS, svc_rbf_best_est_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % svc_rbf_auc)
```


![png](/graph/output_136_0.png)


Now, for kernel sigmoid.


```python
rs_sigmoid.fit(xtrainS, ytrainS)
```




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
              estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
              fit_params=None, iid='warn', n_iter=500, n_jobs=4,
              param_distributions={'kernel': ['sigmoid'], 'gamma': [0.01, 0.001, 0.0001, 1e-05], 'C': [0.001, 0.1, 0.1, 10, 25, 50, 100, 1000]},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='roc_auc', verbose=0)




```python
svc_sigmoid = rs_sigmoid.best_estimator_
```


```python
svc_sigmoid.fit(xtrainS, ytrainS)
```




    SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='sigmoid',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
svc_sigmoid_pred = svc_sigmoid.predict(xtestS)
svc_sigmoid_pred_proba = svc_sigmoid.predict_proba(xtestS)[:, 1]
```

AUC:


```python
svc_sigmoid_auc = roc_auc_score(ytestS, svc_sigmoid_pred_proba)
svc_sigmoid_auc
```




    0.9618103369215271



Recall:


```python
svc_sigmoid_recall = recall_score(ytestS, svc_sigmoid_pred)
svc_sigmoid_recall
```




    0.8014705882352942




```python
fpr, tpr, threshold = roc_curve(ytestS, svc_sigmoid_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % svc_sigmoid_auc)
```


![png](/graph/output_146_0.png)


Let's check for Linear kernel.


```python
rs_linear.fit(xtrainS, ytrainS)
```




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
              estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
              fit_params=None, iid='warn', n_iter=500, n_jobs=4,
              param_distributions={'kernel': ['linear'], 'C': [0.001, 0.1, 0.1, 10, 25, 50, 100, 1000]},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='roc_auc', verbose=0)




```python
svc_linear = rs_linear.best_estimator_
```


```python
svc_linear.fit(xtrainS, ytrainS)
```




    SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False)



Getting the predictions and probabilities.


```python
svc_linear_pred = svc_linear.predict(xtestS)
svc_linear_pred_proba = svc_linear.predict_proba(xtestS)[:, 1]
```

AUC and ROC Curve


```python
svc_linear_auc = roc_auc_score(ytestS, svc_linear_pred_proba)

fpr, tpr, threshold = roc_curve(ytestS, svc_linear_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % svc_linear_auc)
```


![png](/graph/output_154_0.png)


Let's check the recall too.


```python
svc_linear_recall = recall_score(ytestS, svc_linear_pred)
svc_linear_recall
```




    0.6102941176470589



AUC is great in case of a linear kernel however it's less than that of rbf kernel. And its recall decreased quite a bit.

### Naive Bayes Algorithm

Now, let's try the famous Naive Bayes Machine Learning Algorithm.


```python
from sklearn.naive_bayes import GaussianNB
```


```python
nb = GaussianNB()
```

Let's first train the algorithm on the default settings.


```python
nb.fit(xtrain, ytrain)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
nb_pred = nb.predict(xtest)
nb_pred_proba = nb.predict_proba(xtest)[:, 1]
```


```python
nb_auc = roc_auc_score(ytest, nb_pred)
```


```python
fpr, tpr, threshold = roc_curve(ytestS, nb_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % nb_auc)
```


![png](/graph/output_166_0.png)



```python
nb_recall = recall_score(ytest, nb_pred)
nb_recall
```




    0.6617647058823529



Conclusion: Naive Bayes didn't perform well as compared to the other ones.
