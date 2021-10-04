import pandas as pd

df1 = pd.read_csv('ppg_jp_backtrans1.csv')
df2 = pd.read_csv('ppg_jp_backtrans2.csv')
df3 = pd.read_csv('ppg_jp_backtrans3.csv')

df2['id'] = df2['id'].apply(lambda x: x+12500)
df3['id'] = df3['id'].apply(lambda x: x+30000)

df = pd.concat([df1, df2, df3])

df.to_csv('ppg_jp_backtrans.csv', index=False)