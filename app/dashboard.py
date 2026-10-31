"""VRDet dashboard (Streamlit): Risk Radar (production model, live), Model Lab (every model, on demand) and
Stock Lab (train the models on any NSE stock).

Local only: .streamlit/config.toml binds the server to 127.0.0.1 and disables usage statistics.

Usage (from the repo root, with app/runner.py running):
    streamlit run app/dashboard.py
    streamlit run app/dashboard.py -- --replay     # view a session replayed by `runner.py --replay`
"""
import streamlit as st

import lab_page
import radar_page
import stock_page

st.set_page_config(page_title='VRDet Risk Radar', page_icon=':material/radar:', layout='wide')

page = st.navigation([st.Page(radar_page.render, title='Risk Radar', icon=':material/radar:', default=True),
                      st.Page(lab_page.render, title='Model Lab', icon=':material/science:', url_path='lab'),
                      st.Page(stock_page.render, title='Stock Lab', icon=':material/candlestick_chart:',
                              url_path='stocks')])
with st.sidebar:
    st.caption('NIFTY 50 intraday volatility regime from 15-minute bars.')
    st.slider('Sessions shown', 1, 10, 3, key='n_sessions')
page.run()
with st.sidebar:
    st.divider()
    st.caption('Risk indicator only. Not investment advice.')
