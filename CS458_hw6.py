import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import date, timedelta

def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum(xi*yi for xi,yi in zip(X, Y)) - n * xbar * ybar
    denum = sum(xi**2 for xi in X) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b

def plot_fragility(ticker):
    SP_500_ticker = yf.Ticker('^GSPC')
    SP_500 = SP_500_ticker.history(period="22y")
    SP_500_filtered = moving_average(SP_500['Close'], 20)
    SP_500_velocity = np.gradient(SP_500_filtered)
    
    test_ticker = yf.Ticker(ticker)
    test_name = test_ticker.info['shortName']
    test = test_ticker.history(period="22y")
    test_filtered = moving_average(test['Close'], 20)
    test_velocity = np.gradient(test_filtered)
    
    if len(SP_500_velocity) > len(test_velocity):
        test_velocity = np.pad(test_velocity, (0, len(SP_500_velocity) - len(test_velocity)))
    elif len(SP_500_velocity) < len(test_velocity):
        SP_500_velocity = np.pad(SP_500_velocity, (0, len(test_velocity) - len(SP_500_velocity)))

    a, b = best_fit(SP_500_velocity, test_velocity)
    fit = [a + b * xi for xi in SP_500_velocity]
    
    dates = np.linspace(2001, 2023, test_filtered.size)

    plt.plot(dates, test_filtered)
    plt.title(test_name + " Filtered")
    plt.xlabel("Date")
    plt.show()
    
    plt.scatter(SP_500_velocity[-test_velocity.size:], test_velocity)
    plt.plot(SP_500_velocity, fit, color='red', linestyle='--', linewidth=2)
    plt.title(test_name + " Fragility")
    plt.xlabel("S&P 500 Velocity")
    plt.ylabel(test_name + " Velocity")
    plt.show()
    
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    
    fragility_index = sum(np.multiply(abs(SP_500_velocity), test_velocity)) / test_velocity.size
    print(test_name + ' Fragility index = ' + str(fragility_index))
    
    return fragility_index

def analyze_portfolio(tickers):
    fragilities = []
    for ticker in tickers:
        fragilities.append(plot_fragility(ticker))        
    print()
    average_fragility = 0
    for i in range(len(tickers)):
        average_fragility += fragilities[i]
        ticker_ticker = yf.Ticker(tickers[i])
        ticker_name = ticker_ticker.info['shortName']
        print(ticker_name + " Fragility Index = " + str(fragilities[i]))
    average_fragility / len(fragilities)
    print()
    print("Portfolio Average Fragility: " + str(average_fragility))

# Analyze High Antifragility Portfolio
antifragile_tickers = ['AZO', 'RTX', 'CHTR', 'META', 'TJX', 'NVDA']
analyze_portfolio(antifragile_tickers)