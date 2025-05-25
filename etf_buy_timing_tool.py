import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import yfinance as yf

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
    today = datetime.datetime.today().strftime("%Y%m%d")
    one_month_ago = (datetime.datetime.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    url = f"https://api.finance.naver.com/siseJson.naver?symbol={ticker}&requestType=1&startTime={one_month_ago}&endTime={today}&timeframe=day"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    text = res.text.strip().replace('\n', '')
    rows = eval(text)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df['날짜'] = pd.to_datetime(df['날짜'])
    df.set_index('날짜', inplace=True)
    df = df.rename(columns={"종가": "Close", "시가": "Open", "고가": "High", "저가": "Low"})
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)
    return df[['Open', 'High', 'Low', 'Close']]

# Streamlit UI 설정
st.title("ETF 매수 타이밍 분석기")
st.write("최근 30일간 기술적 지표를 분석하여 지금이 매수 타이밍인지 알려드립니다.")

etfs = {
    "KODEX S&P500": "KODEX",
    "QQQM": "QQQM",
    "SPLG": "SPLG",
    "SCHD": "SCHD"
}

selected_etf = st.selectbox("ETF 종목을 선택하세요:", list(etfs.keys()))
ticker = etfs[selected_etf]

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=30)

# 데이터 불러오기
if selected_etf == "KODEX S&P500":
    data = get_korean_stock_price("379800")
else:
    raw = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(-1)
    data = raw[['Open', 'High', 'Low', 'Close']].copy()
    data = data.dropna()
    data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)

if data.empty:
    st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
    st.stop()

data['RSI'] = compute_rsi(data['Close'], period=14)
data['MACD'], data['MACD_signal'] = compute_macd(data['Close'])
data['MA20'] = data['Close'].rolling(window=20).mean()
data['STD20'] = data['Close'].rolling(window=20).std()
data['Lower_BB'] = data['MA20'] - 2 * data['STD20']

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

st.subheader(f"{selected_etf} 분석 결과")
st.write(f"- RSI: {data['RSI'].iloc[-1]:.2f} ({'충족' if rsi_cond else '비충족'})")
st.write(f"- MACD > Signal: {macd_cond}")
st.write(f"- 현재가 < MA20 * 0.98: {ma20_cond}")
st.write(f"- 현재가 ≦ 볼린저밴드 하단 +5%: {bb_cond}")

if true_count >= 2:
    st.success("✅ 지금은 매수 타이밍일 수 있습니다. (2개 이상 조건 충족)")
    if selected_etf == "SCHD":
        st.info("**SCHD는 기술적 조정과 고배당 특성으로 인해 현재 가격이 특히 매력적인 매수 구간입니다. 장기 보유 + 배당 전략에 적합합니다.**")
else:
    st.warning("❌ 아직 매수 타이밍으로 보기 어렵습니다.")

st.subheader("최근 30일간 가격 데이터")
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    close_data = data.copy()
    ax.plot(close_data.index, close_data['Close'], label='Close')
    ax.set_ylim(close_data['Close'].min() * 0.95, close_data['Close'].max() * 1.05)
    ax.set_title(f'{selected_etf} 종가 추이')
    ax.set_ylabel('가격')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"📉 차트 표시 중 오류 발생: {e}")

st.dataframe(data.tail(10))
