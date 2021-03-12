import pandas as pd 
import pandas_datareader as pdr
import matplotlib.pyplot as mpl

from fbprophet import Prophet
from datetime import datetime, timedelta, date

koreaStock = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]
koreaStock = koreaStock.rename(columns={'회사명':'name','종목코드':'stockCode','상장일':'stockBirth'})
koreaStock = koreaStock[['name', 'stockCode','stockBirth']]
userSerchingStock = input()

stockCode = koreaStock[koreaStock['name'].isin([userSerchingStock])].iloc[0,1]
stockCode = str(stockCode).zfill(6) + '.KS'

stockBirth = koreaStock[koreaStock['name'].isin([userSerchingStock])].iloc[0,2]
stockBirth = str(stockBirth).split('-')
print(stockBirth[0],stockBirth[1],stockBirth[2])

Nyear = datetime.today().year
Nmonth = datetime.today().month
Nday = datetime.today().day
yesterday = (date.today() - timedelta(1)).strftime('%Y-%m-%d')

start = datetime(int(stockBirth[0]), int(stockBirth[1]), int(stockBirth[2]))
done = datetime(Nyear,Nmonth,Nday)

APPLE = pdr.DataReader(stockCode, "yahoo", start, done)
realityDF = pd.DataFrame({'ds': APPLE.index, 'y' : APPLE['Close']})

model = Prophet()
model.fit(realityDF)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

futureData = forecast[forecast['ds'].isin([yesterday])]
futureData = futureData.iloc[0,18]

todayData = realityDF.loc[yesterday : yesterday]
todayData = todayData.iloc[0,1]

ErrorValue = todayData - futureData
dataSet = str(yesterday) + '  ' + str(ErrorValue) 
print(dataSet + '\n')
print("nowData", todayData,'원', ' | ', "futureData", futureData,'원') 
model.plot(forecast) 
#model.plot_components(forecast)
mpl.show()



