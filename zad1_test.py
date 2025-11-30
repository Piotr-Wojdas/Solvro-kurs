from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


def fillna_with_distribution(df, columns):
	for col in columns:
		# value_counts tylko na podstawie X_train
		value_counts = X_train[col].value_counts(normalize=True, dropna=True)
		missing = df[col].isna()
		if missing.sum() > 0:
			filled_values = np.random.choice(
				value_counts.index,
				size=missing.sum(),
				p=value_counts.values
			)
			df.loc[missing, col] = filled_values
	return df

# usuwanie pustych rekordów
X = fillna_with_distribution(X, ['job', 'education', 'contact'])
mask = (X['poutcome'].isna())
X.loc[mask, 'poutcome'] = 'none'
X['is_first_time'] = (X['pdays'] == -1).astype(int)


# Zamień -1 na np.nan w pdays
for df in [X_train, X_val, X_test, X]:
    df['pdays'] = df['pdays'].replace(-1, np.nan)

median_pdays = X_train.loc[X_train['pdays'].notna(), 'pdays'].median()
X.loc[X['pdays'].isna(), 'pdays'] = median_pdays
X['pdays'] = X['pdays'].astype(int)

# One-hot encoding 
dummy_cols = ['job', 'marital','poutcome']
X = pd.get_dummies(X, columns=dummy_cols, drop_first=False)

for col in ['default', 'housing', 'loan','contact']:
    if col == 'contact':
        X[col] = X[col].map({'telephone': 1, 'cellular': 0})
    else:
        X[col] = X[col].map({'yes': 1, 'no': 0})

# label encoding
enc_education = LabelEncoder()
X['education'] = enc_education.fit_transform(X['education'])


# Kodowanie cykliczne dla kolumny 'month'
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_to_num = {month: idx for idx, month in enumerate(months)}
X['month_num'] = X['month'].map(month_to_num)
X['month_sin'] = np.sin(2 * np.pi * X['month_num'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month_num'] / 12)
X = X.drop(columns=['month', 'month_num'])

# Wizualizacja macierzy korelacji za pomocą seaborn

plt.figure(figsize=(16, 12))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Macierz korelacji między cechami")
plt.show()



