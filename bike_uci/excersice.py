import pandas as pd

df = pd.read_csv("hour.csv")

print(df[['temp', 'hum','windspeed']].agg(['max', 'min', 'mean', 'std', 'median']))