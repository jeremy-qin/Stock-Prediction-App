import streamlit as st
from datetime import date
from subprocess import call
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START_DATE = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout='wide')
st.sidebar.markdown("<div><img src='https://cdn-0.emojis.wiki/emoji-pics/microsoft/chart-increasing-microsoft.png' width=100 /><h1 style='display:inline-block'>Stock Analytics</h1></div>", unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.markdown("This dashboard allows you to analyse trending 📈 stocks using Python and Streamlit.")
st.sidebar.markdown("")
st.sidebar.markdown("It also allows you to see prediction of stock prices for a specific period of time using Facebook Prophet package.")
st.title("Stock Prediction App")

selected_stocks = st.text_input("Search for a stock", value="")
n_years = st.slider("Years of prediction:",1,3)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data["Date"], y=data["High"], name="stock_high"))
    fig2.layout.update(title_text="Time Series Data for Highest Value", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

if st.button("Get Data") and selected_stocks != "":
    data_load_state = st.text("Load data...")
    data = load_data(selected_stocks)
    data_load_state.text("Loading data...done")

    if data.isnull:
        st.write("Please enter a correct stock's name")
        
    else:

        st.subheader("Raw data")
        st.write(data.tail())
        plot_raw_data()

        #Forecasting
        df_train = data[["Date", "Close"]]
        df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})

        m = Prophet()
        m.fit(df_train)

        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader("Forecast Data")
        st.write(forecast.tail())

        fig1 = plot_plotly(m,forecast)
        st.plotly_chart(fig1)

        st.subheader("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
else:
    st.write("Please write a stock name")
