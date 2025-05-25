import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
import ast

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

if selected_etf == "KODEX S&P500":
    data = get_korean_stock_price("379800")
else:
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close']].dropna().astype(float)

if data.empty:
    st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
    st.stop()

# 기술적 지표 계산
data['RSI'] = compute_rsi(data['Close'], period=14)
data['MACD'], data['MACD_signal'] = compute_macd(data['Close'])
data['MA20'] = data['Close'].rolling(window=20).mean()
data['STD20'] = data['Close'].rolling(window=20).std()
data['Lower_BB'] = data['MA20'] - 2 * data['STD20']

# 필요한 컬럼 모두 생성되어 있는지 확인 (없으면 np.nan으로 채움)
import numpy as np
required_cols = ['RSI', 'MACD', 'MACD_signal', 'MA20', 'STD20', 'Lower_BB']
for col in required_cols:
    if col not in data.columns:
        data[col] = np.nan

# 결측치 제거 후 마지막 행 추출
filtered = data.dropna(subset=required_cols)
if filtered.empty:
    st.error("기술적 지표 계산을 위한 데이터가 부족합니다. 30일치 이상 가격 데이터가 필요합니다.")
    st.stop()
latest = filtered.iloc[-1]

# 매수 조건
rsi_cond = latest['RSI'] < 40
macd_cond = latest['MACD'] > latest['MACD_signal']
ma20_cond = latest['Close'] < latest['MA20'] * 0.98
bb_cond = latest['Close'] <= latest['Lower_BB'] * 1.05

true_count = sum([rsi_cond, macd_cond, ma20_cond, bb_cond])

st.subheader(f"{selected_etf} 분석 결과")
st.write(f"- RSI: {latest['RSI']:.2f} ({'충족' if rsi_cond else '비충족'})")
st.write(f"- MACD > Signal: {macd_cond}")
st.write(f"- 현재가 < MA20 * 0.98: {ma20_cond}")
st.write(f"- 현재가 ≦ 볼린저밴드 하단 +5%: {bb_cond}")

if true_count >= 2:
    st.success("✅ 지금은 매수 타이밍일 수 있습니다. (2개 이상 조건 충족)")
    if selected_etf == "SCHD":
        st.info("**SCHD는 기술적 조정과 고배당 특성으로 인해 현재 가격이 특히 매력적인 매수 구간입니다. 장기 보유 + 배당 전략에 적합합니다.**")
else:
    st.warning("❌ 아직 매수 타이밍으로 보기 어렵습니다.")

# 캔들차트
st.subheader("최근 30일간 캔들차트")
add_plots = []
if 'MA20' in data.columns and not data['MA20'].dropna().empty:
    add_plots.append(mpf.make_addplot(data['MA20'].dropna(), color='orange', width=1.2))
if 'Lower_BB' in data.columns and not data['Lower_BB'].dropna().empty:
    add_plots.append(mpf.make_addplot(data['Lower_BB'].dropna(), color='blue', linestyle='--', width=1.0))

fig, _ = mpf.plot(
    data,
    type='candle',
    style='charles',
    mav=(20,),
    volume=False,
    addplot=add_plots,
    show_nontrading=True,
    datetime_format='%Y-%m-%d',
    xrotation=45,
    returnfig=True,
    figsize=(10, 6),
)
st.pyplot(fig)
