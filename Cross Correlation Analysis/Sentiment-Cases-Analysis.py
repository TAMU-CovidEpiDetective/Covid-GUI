import pandas as pd
import scipy.stats as stats

df = pd.read_csv('Cases-Sentiment-Analysis.csv')
overall_pearson_r = df.corr().iloc[0,1]
print(f"Pandas computed Pearson r: {overall_pearson_r}")


r, p = stats.pearsonr(df.dropna()['Sentiment'], df.dropna()['Cases'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")


for i in range (1,7):
    ff = df.dropna()['Sentiment'].tail(-i)
    gg = df.dropna()['Cases'].iloc[:-i]
    f = pd.Series(ff).rolling(3).std()
    a = f.diff()
    g = pd.Series(gg).rolling(3).std()
    b = g.diff()
    r, p = stats.pearsonr(a.dropna(), b.dropna())
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")



