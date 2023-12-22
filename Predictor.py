import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yahoofinancials import YahooFinancials
import mpl_finance
import matplotlib.pyplot as plt
import datetime

stock = input("Which stock?")

highs = []
lows = []
opens = []
closes = []

yahoo_financials = YahooFinancials(str(stock))
stats = (yahoo_financials.get_historical_price_data("2018-01-01", "2021-03-30", "daily"))

i = 0
for date in stats[str(stock)]["prices"]:
	if i == 0:
		i += 1
		continue
	highs.append(date["high"])
	lows.append(date["low"])
	opens.append(date["open"])
	closes.append(date["close"])
	i += 1
print("No. of data pts:",i)

total = []
totalopens = []
for j in range(4):
	opens.append(0)

for i in range(i-1):
	total.append([opens[i], lows[i], highs[i], closes[i]])
	#row=[]
	#for p in range(4):
	   #row.append(opens[i-p])
	#totalopens.append(row)   

#for totalopen in totalopens:
#print(totalopens)

#months = int(input("How many months for validation?"))

def Predictor(lst,months):
	total_training = lst[0:i-months]
	total_validation = lst[i-months:]

	df = pd.DataFrame(total_training, dtype=float)
	XTrain = df.iloc[:, :-1]
	yTrain = df.iloc[:, [-1]]

	clf = LinearRegression()
	clf.fit(XTrain, yTrain)

	print("\n\n")
	
	dfp = pd.DataFrame(total_validation, dtype=float)
	XValidation = dfp.iloc[:, :-1]
	YValidation = dfp.iloc[:, [-1]]

	print("\n\nPrediction")
	YPrediction = clf.predict(XValidation)

	print("\nOriginal")
	print(YValidation)

	print("\nPredicted")
	print(YPrediction)

Predictor(total,1)

