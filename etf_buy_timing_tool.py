import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import ast

# 기술 지표 계산 함수들
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# 네이버에서 국내 주식 데이터 가져오기
def get_korean_stock_price(ticker):
    today = datetime.datetime.today().strftime("%Y%m%d")
    one_month_ago = (datetime.datetime.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    url = f"https://api.finance.naver.com/siseJson.naver?symbol={ticker}&requestType=1&startTime={one_month_ago}&endTime={today}&timeframe=day"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    text = res.text.strip().replace('\n', '')
    try:
        rows = ast.literal_eval(text)
    except Exception as e:
        st.error(f"데이터 파싱 오류: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df['날짜'] = pd.to_datetime(df['날짜'])
    df.set_index('날짜', inplace=True)
    df = df.rename(columns={"종가": "Close", "시가": "Open", "고가": "High", "저가": "Low"})
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['Open', 'High', 'Low', 'Close']]

# Streamlit UI
st.title("ETF 매수 타이밍 판별기")

etfs = {
    "KODEX S&P500": "KODEX",
    "QQQM": "QQQM",
    "SPLG": "SPLG",
    "SCHD": "SCHD"
}

selected_etf = st.selectbox("ETF 종목을 선택하세요:", list(etfs.keys()))
ticker = etfs[selected_etf]

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=90)

if selected_etf == "KODEX S&P500":
    data = get_korean_stock_price("379800")
else:
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error("해외 ETF 데이터를 가져올 수 없습니다.")
        st.stop()
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

data['RSI'] = compute_rsi(data['Close'], period=14)
data['MACD'], data['MACD_signal'] = compute_macd(data['Close'])
data['MA20'] = data['Close'].rolling(window=20).mean()
data['STD20'] = data['Close'].rolling(window=20).std()
data['Lower_BB'] = data['MA20'] - 2 * data['STD20']

data = data.dropna(subset=['RSI', 'MACD', 'MACD_signal', 'MA20', 'STD20', 'Lower_BB'])

if data.empty:
    st.warning("최근 데이터가 부족해 분석할 수 없습니다.")
    st.stop()

latest = data.iloc[-1]

rsi_cond = latest['RSI'] < 40
macd_cond = latest['MACD'] > latest['MACD_signal']
ma20_cond = latest['Close'] < latest['MA20'] * 0.98
bb_cond = latest['Close'] <= latest['Lower_BB'] * 1.05

true_count = sum([rsi_cond, macd_cond, ma20_cond, bb_cond])

st.subheader(f"{selected_etf} 매수 타이밍 판별 결과")
st.write(f"- RSI: {latest['RSI']:.2f} ({'충족' if rsi_cond else '비충족'})")
st.write(f"- MACD > Signal: {macd_cond}")
st.write(f"- 현재가 < MA20 * 0.98: {ma20_cond}")
st.write(f"- 현재가 ≦ 볼린저밴드 하단 +5%: {bb_cond}")

if true_count >= 2:
    st.success("✅ 지금은 매수 타이밍일 수 있습니다. (2개 이상 조건 충족)")
else:
    st.warning("❌ 아직 매수 타이밍으로 보기 어렵습니다.")
