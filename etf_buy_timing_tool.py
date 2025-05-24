import streamlit as st
import yfinance as yf
import pandas as pd
import talib
import datetime
import matplotlib.pyplot as plt

# Streamlit UI 설정
st.title("ETF 매수 타이밍 분석기")
st.write("최근 30일간 기술적 지표를 분석하여 지금이 매수 타이밍인지 알려드립니다.")

# 종목 리스트
etfs = {
    "KODEX S&P500": "379800.KQ",
    "QQQM": "QQQM",
    "SPLG": "SPLG",
    "SCHD": "SCHD"
}

selected_etf = st.selectbox("ETF 종목을 선택하세요:", list(etfs.keys()))
ticker = etfs[selected_etf]

# 오늘 날짜 기준 최근 30일 데이터
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=30)

# 데이터 가져오기
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
    st.stop()

# 기술적 지표 계산
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'])
data['MA20'] = data['Close'].rolling(window=20).mean()
data['Lower_BB'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()

# 매수 조건
rsi_cond = data['RSI'].iloc[-1] < 40
macd_cond = data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]
ma20_cond = data['Close'].iloc[-1] < data['MA20'].iloc[-1] * 0.98
bb_cond = data['Close'].iloc[-1] <= data['Lower_BB'].iloc[-1] * 1.05

true_count = sum([rsi_cond, macd_cond, ma20_cond, bb_cond])

# 결과 표시
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

# 차트 표시
st.subheader("가격 차트 및 이평선")
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label='Close')
ax.plot(data.index, data['MA20'], label='MA20')
ax.plot(data.index, data['Lower_BB'], label='Lower BB')
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# 데이터 테이블
st.subheader("기술적 지표 테이블")
st.dataframe(data.tail(10))
