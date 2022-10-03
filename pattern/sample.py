import numpy as np
import pandas as pd

df = pd.DataFrame(data={'col1': [1, np.nan, 2, np.nan], 'col2': [1, 0, 0, 1]})
df['col1'].fillna(np.mean(df['col1']), inplace=True)

train = df[:3]
test = df[3:]
