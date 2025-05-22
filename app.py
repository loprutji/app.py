
import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽")

st.title("⚽ Premier League Match Predictor")

# ดึง API Key อย่างถูกต้อง
API_KEY = st.secrets["FOOTBALL_API_KEY"]
headers = {'X-Auth-Token': API_KEY}
url = 'https://api.football-data.org/v4/competitions/PL/matches?season=2024'

@st.cache_data(ttl=3600)
def fetch_matches():
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    data = response.json()['matches']
    played = [m for m in data if m['score']['fullTime']['home'] is not None]
    return played

matches = fetch_matches()

if not matches:
    st.error("ไม่สามารถโหลดข้อมูลการแข่งขันได้")
else:
    df = pd.DataFrame([{
        'home_team': m['homeTeam']['name'],
        'away_team': m['awayTeam']['name'],
        'home_goals': m['score']['fullTime']['home'],
        'away_goals': m['score']['fullTime']['away']
    } for m in matches])

    df['result'] = df.apply(lambda row: 1 if row['home_goals'] > row['away_goals'] else (-1 if row['home_goals'] < row['away_goals'] else 0), axis=1)
    df['goal_diff'] = df['home_goals'] - df['away_goals']

    X = df[['goal_diff']]
    y = df['result']
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X, y)

    st.subheader("🔮 ทำนายผลการแข่งขัน")
    home_team = st.selectbox("เลือกทีมเจ้าบ้าน", df['home_team'].unique())
    away_team = st.selectbox("เลือกทีมเยือน", df['away_team'].unique())

    if st.button("คาดการณ์ผล"):
        recent_home = df[df['home_team'] == home_team].tail(5)['goal_diff'].mean()
        recent_away = df[df['away_team'] == away_team].tail(5)['goal_diff'].mean()
        test_diff = recent_home - (-recent_away)

        pred = model.predict([[test_diff]])[0]
        label = {1: "🏠 เจ้าบ้านชนะ", 0: "⚖ เสมอ", -1: "🛫 ทีมเยือนชนะ"}[pred]
        st.success(f"ผลคาดการณ์: {label}")
