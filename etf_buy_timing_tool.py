import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import requests

# RSI 계산 함수
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD 계산 함수
def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Naver Finance로부터 국내 주식 시세 불러오기
def get_korean_stock_price(ticker):
    url = f"https://api.finance.naver.com/siseJson.naver?symbol={ticker}&requestType=1&startTime=20240401&endTime=20240524&timeframe=day"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    text = res.text.strip().replace('\n', '')
    rows = eval(text)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df['날짜'] = pd.to_datetime(df['날짜'])
    df.set_index('날짜', inplace=True)
    df = df.rename(columns={"종가": "Close"})
    df['Close'] = pd.to_numeric(df['Close'])
    return df[['Close']]

# Streamlit UI 설정
st.title("ETF 매수 타이밍 분석기")
st.write("최근 30일간 기술적 지표를 분석하여 지금이 매수 타이밍인지 알려드립니다.")

# 종목 리스트
etfs = {
    "KODEX S&P500": "KODEX",  # 국내 종목 코드 별도 처리
    "QQQM": "QQQM",
    "SPLG": "SPLG",
    "SCHD": "SCHD"
}

selected_etf = st.selectbox("ETF 종목을 선택하세요:", list(etfs.keys()))
ticker = etfs[selected_etf]

# 날짜 설정
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=30)

# 데이터 불러오기
if selected_etf == "KODEX S&P500":
    data = get_korean_stock_price("379800")
else:
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
    st.stop()

# 기술적 지표 계산
data['RSI'] = compute_rsi(data['Close'], period=14)
data['MACD'], data['MACD_signal'] = compute_macd(data['Close'])
data['MA20'] = data['Close'].rolling(window=20).mean()
data['STD20'] = data['Close'].rolling(window=20).std()
data['Lower_BB'] = data['MA20'] - 2 * data['STD20']

# 매수 조건
rsi_cond = data['RSI'].iloc[-1] < 40
macd_cond = data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]
ma20_cond = data['Close'].iloc[-1] < data['MA20'].iloc[-1] * 0.98
bb_cond = data['Close'].iloc[-1] <= data['Lower_BB'].iloc[-1] * 1.05

true_count = sum([
    int(rsi_cond),
    int(macd_cond),
    int(ma20_cond),
    int(bb_cond)
])
