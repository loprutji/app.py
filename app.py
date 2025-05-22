import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽")

st.title("⚽ Premier League Match Predictor")

# API Key
FOOTBALL_API_KEY = st.secrets["FOOTBALL_API_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

# -------------------- ข้อมูลผลแข่ง ----------------------
@st.cache_data(ttl=3600)
def fetch_matches():
    url = 'https://api.football-data.org/v4/competitions/PL/matches?season=2024'
    headers = {'X-Auth-Token': FOOTBALL_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    data = response.json().get('matches', [])
    played = [m for m in data if m['score']['fullTime']['home'] is not None]
    return played

@st.cache_data(ttl=3600)
def fetch_fixtures():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    headers = {'X-Auth-Token': FOOTBALL_API_KEY}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            return []
        return r.json().get("matches", [])
    except:
        return []

# -------------------- ตารางคะแนน ----------------------
@st.cache_data(ttl=3600)
def fetch_standings():
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {'X-Auth-Token': FOOTBALL_API_KEY}
    r = requests.get(url, headers=headers)
    data = r.json()
    if 'standings' not in data or not data['standings']:
        return pd.DataFrame([{"ทีม": "⚠️ ไม่พบข้อมูล", "แต้ม": "-"}])
    table = data['standings'][0]['table']
    return pd.DataFrame([{
        'อันดับ': t['position'],
        'ทีม': t['team']['name'],
        'แข่ง': t['playedGames'],
        'ชนะ': t['won'],
        'เสมอ': t['draw'],
        'แพ้': t['lost'],
        'ได้': t['goalsFor'],
        'เสีย': t['goalsAgainst'],
        'แต้ม': t['points']
    } for t in table])

# -------------------- ราคาต่อรอง Handicap ----------------------
@st.cache_data(ttl=3600)
def fetch_odds():
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'eu',
        'markets': 'spreads',  # Handicap
        'oddsFormat': 'decimal'
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return []
    return r.json()

# -------------------- ส่วน UI ----------------------
tab1, tab2, tab3 = st.tabs(["🔮 คาดการณ์", "📊 ตารางคะแนน", "💸 ราคาต่อรอง"])

# 🔮 คาดการณ์ผล
with tab1:
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

        st.subheader("เลือกคู่การแข่งขัน")
        home_team = st.selectbox("ทีมเจ้าบ้าน", df['home_team'].unique())
        away_team = st.selectbox("ทีมเยือน", df['away_team'].unique())

        if st.button("คาดการณ์ผล"):
            recent_home = df[df['home_team'] == home_team].tail(5)['goal_diff'].mean()
            recent_away = df[df['away_team'] == away_team].tail(5)['goal_diff'].mean()
            test_diff = recent_home - (-recent_away)

            pred = model.predict([[test_diff]])[0]
            label = {1: "🏠 เจ้าบ้านชนะ", 0: "⚖ เสมอ", -1: "🛫 ทีมเยือนชนะ"}[pred]
            st.success(f"ผลคาดการณ์: {label}")

        fixtures = fetch_fixtures()
        if not fixtures:
            st.warning("❌ ไม่สามารถโหลดข้อมูลการแข่งขันได้")
        else:
            st.subheader("📅 นัดที่กำลังจะมาถึง")
            for m in fixtures[:5]:
                utc_time = datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
                local_time = utc_time + timedelta(hours=7)
                st.markdown(f"**{m['homeTeam']['name']} vs {m['awayTeam']['name']}**")
                st.write("🕓", local_time.strftime("%d/%m/%Y %H:%M"))
                st.divider()

# 📊 ตารางคะแนน
with tab2:
    st.subheader("📊 ตารางคะแนนพรีเมียร์ลีก")
    standings_df = fetch_standings()
    st.dataframe(standings_df, use_container_width=True)

# 💸 ราคาต่อรอง Handicap
with tab3:
    st.subheader("💸 ราคาต่อรอง Handicap (Asian)")
    odds = fetch_odds()
    if not odds:
        st.warning("ไม่สามารถโหลดราคาต่อรองจาก API ได้")
    else:
        for match in odds[:10]:
            home = match.get("home_team")
            away = match.get("away_team")
            bookies = match.get("bookmakers", [])
            if not bookies:
                continue
            markets = bookies[0].get("markets", [])
            if not markets:
                continue
            outcomes = markets[0].get("outcomes", [])
            st.markdown(f"**{home} vs {away}**")
            for o in outcomes:
                name = o.get("name", "-")
                point = o.get("point", "-")
                price = o.get("price", "-")
                st.write(f"{name} ({point}): {price}")
            st.divider()
