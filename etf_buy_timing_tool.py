import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
import ast

# 한글 폰트 설정 (matplotlib에서 캔들차트 한글 깨짐 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
# 해외 ETF는 넉넉하게 데이터를 요청하고 최근 30 거래일만 자름
start_date = end_date - datetime.timedelta(days=90)

if selected_etf == "KODEX S&P500":
    data = get_korean_stock_price("379800")
else:
    import yfinance as yf
    raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if raw_data.empty:
        st.error("해외 ETF 데이터를 가져올 수 없습니다.")
        st.stop()
    raw_data = raw_data[['Open', 'High', 'Low', 'Close']].dropna()
    raw_data[['Open', 'High', 'Low', 'Close']] = raw_data[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce')
    raw_data = raw_data.dropna()
    # 기술적 지표 계산 (전체 데이터 기준)
    raw_data['RSI'] = compute_rsi(raw_data['Close'], period=14)
    raw_data['MACD'], raw_data['MACD_signal'] = compute_macd(raw_data['Close'])
    raw_data['MA20'] = raw_data['Close'].rolling(window=20).mean()
    raw_data['STD20'] = raw_data['Close'].rolling(window=20).std()
    raw_data['Lower_BB'] = raw_data['MA20'] - 2 * raw_data['STD20']
    full_cols = ['RSI', 'MACD', 'MACD_signal', 'MA20', 'STD20', 'Lower_BB']
    # 존재하는 컬럼만 필터링
    safe_cols = [col for col in full_cols if col in raw_data.columns and raw_data[col].notna().any()]
    missing_cols = [col for col in full_cols if col not in raw_data.columns]
    if not safe_cols:
        st.error(f"기술적 지표 컬럼이 누락되었거나 NaN 값만 존재합니다. 누락된 컬럼: {missing_cols}")
        st.stop()
    try:
        valid_data = raw_data.dropna(subset=safe_cols)
    except KeyError as e:
        st.error(f"기술적 지표 컬럼이 누락되어 분석할 수 없습니다: {e}")
        st.stop()
    data = valid_data.tail(60)

if data.empty:
    st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
    st.stop()

# 이미 계산된 기술적 지표가 있으므로 생략

# 유효한 컬럼만 필터링하여 dropna에 사용
required_cols = ['RSI', 'MACD', 'MACD_signal', 'MA20', 'STD20', 'Lower_BB']
existing_cols = []
for col in required_cols:
    if col in data.columns and data[col].notna().any():
        existing_cols.append(col)

if not existing_cols:
    st.error("기술적 지표 계산에 필요한 데이터가 없습니다.")
    st.stop()  # 경고 발생 방지를 위해 st.stop()은 단독 실행

# dropna에서 오류가 발생하지 않도록 복사본을 사용
safe_cols = [col for col in existing_cols if col in data.columns and col in data.keys()]
data_filtered = data.copy()
try:
        filtered = data_filtered.dropna(subset=safe_cols)
except KeyError as e:
    st.error(f"기술적 지표 컬럼이 누락되어 분석할 수 없습니다: {e}")
    st.stop()
if filtered.empty:
    st.warning("기술적 지표 계산을 위한 데이터가 부족하여 분석을 건너뜁니다.")
    st.stop()

latest = filtered.iloc[-1]

# 매수 조건 평가
rsi_cond = latest['RSI'] < 40
macd_cond = latest['MACD'] > latest['MACD_signal']
ma20_cond = latest['Close'] < latest['MA20'] * 0.98
bb_cond = latest['Close'] <= latest['Lower_BB'] * 1.05

true_count = sum([rsi_cond, macd_cond, ma20_cond, bb_cond])

# 결과 출력
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

# 캔들차트 표시
st.subheader("최근 60일간 캔들차트")
add_plots = []
if 'MA20' in data.columns and not data['MA20'].dropna().empty:
    add_plots.append(mpf.make_addplot(data['MA20'], color='orange', width=1.2))
if 'Lower_BB' in data.columns and not data['Lower_BB'].dropna().empty:
    add_plots.append(mpf.make_addplot(data['Lower_BB'], color='blue', linestyle='--', width=1.0))

# 장이 열린 날만 표시하는 별도 필터는 불필요하며 show_nontrading=True가 처리함  # 장이 열린 날만 유지
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
