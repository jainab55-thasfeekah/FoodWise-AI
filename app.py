import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from io import BytesIO
from datetime import date, timedelta
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_NAME = "foodwise.db"

REQUIRED_COLUMNS = [
    "date", "meal", "food_item", "attendance",
    "prepared_quantity", "consumed_quantity"
]

MODEL_FEATURES = [
    "attendance", "day", "meal", "food_item", "special_event"
]

# --------------------------------------------------
# CUSTOM DARK CHARCOAL + PURPLE THEME
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #111018;
        color: #F4F0FA;
    }

    [data-testid="stSidebar"] {
        background-color: #191624;
        border-right: 1px solid #35264F;
    }

    h1, h2, h3 {
        color: #C9A7FF !important;
    }

    p, label, span {
        color: #E5DDF0;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #211A30, #302044);
        border: 1px solid #65439A;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #D8B9FF;
    }

    .stButton > button {
        background-color: #7946B8;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 9px 18px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #9565D1;
        color: white;
        border: 1px solid #C9A7FF;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #211C2C;
    }

    .hero {
        background: linear-gradient(120deg, #21172F, #392153);
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #7048A4;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin-bottom: 5px;
    }

    .hero p {
        color: #D9C9EB;
        font-size: 16px;
    }

    .section-card {
        background-color: #1D1928;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #423052;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS food_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            meal TEXT,
            food_item TEXT,
            attendance INTEGER,
            prepared_quantity REAL,
            consumed_quantity REAL,
            leftover_quantity REAL,
            special_event TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_records(data):
    conn = sqlite3.connect(DB_NAME)

    for _, row in data.iterrows():
        conn.execute("""
            INSERT INTO food_records (
                date, meal, food_item, attendance,
                prepared_quantity, consumed_quantity,
                leftover_quantity, special_event
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row["date"]),
            str(row["meal"]),
            str(row["food_item"]),
            int(row["attendance"]),
            float(row["prepared_quantity"]),
            float(row["consumed_quantity"]),
            float(row["leftover_quantity"]),
            str(row["special_event"])
        ))

    conn.commit()
    conn.close()


def load_saved_records():
    conn = sqlite3.connect(DB_NAME)
    data = pd.read_sql_query(
        "SELECT * FROM food_records ORDER BY date",
        conn
    )
    conn.close()

    if not data.empty:
        data = data.drop(columns=["id"], errors="ignore")

    return data


initialize_database()


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data(data):
    data = data.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if "special_event" not in data.columns:
        data["special_event"] = "No"

    if "leftover_quantity" not in data.columns:
        data["leftover_quantity"] = (
            pd.to_numeric(data["prepared_quantity"], errors="coerce")
            - pd.to_numeric(data["consumed_quantity"], errors="coerce")
        ).clip(lower=0)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["attendance"] = pd.to_numeric(
        data["attendance"], errors="coerce"
    )
    data["prepared_quantity"] = pd.to_numeric(
        data["prepared_quantity"], errors="coerce"
    )
    data["consumed_quantity"] = pd.to_numeric(
        data["consumed_quantity"], errors="coerce"
    )
    data["leftover_quantity"] = pd.to_numeric(
        data["leftover_quantity"], errors="coerce"
    )

    data = data.dropna(subset=[
        "date", "attendance", "prepared_quantity",
        "consumed_quantity"
    ])

    data["attendance"] = data["attendance"].clip(lower=0)
    data["prepared_quantity"] = data["prepared_quantity"].clip(lower=0)
    data["consumed_quantity"] = data["consumed_quantity"].clip(lower=0)
    data["leftover_quantity"] = data["leftover_quantity"].clip(lower=0)

    data["meal"] = data["meal"].astype(str).str.strip().str.title()
    data["food_item"] = data["food_item"].astype(str).str.strip()
    data["special_event"] = (
        data["special_event"].fillna("No").astype(str).str.strip()
    )

    data["day"] = data["date"].dt.day_name()
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")

    data = data[data["food_item"] != ""]
    data = data[data["meal"] != ""]

    return data.reset_index(drop=True)


# --------------------------------------------------
# SYNTHETIC DEMO DATA
# --------------------------------------------------

def generate_demo_data():
    rng = np.random.default_rng(42)

    food_items = [
        "Vegetable Rice", "Chicken Biryani", "Idli",
        "Dosa", "Chapati", "Pongal"
    ]

    meals = ["Breakfast", "Lunch", "Dinner"]

    records = []
    start = date.today() - timedelta(days=149)

    for day_number in range(150):
        current_date = start + timedelta(days=day_number)
        day_name = current_date.strftime("%A")

        weekend = day_name in ["Saturday", "Sunday"]
        event = "Yes" if day_number % 30 == 0 else "No"

        attendance = int(
            rng.normal(450 if not weekend else 300, 45)
        )
        attendance = max(100, attendance)

        for meal in meals:
            for food in food_items:
                popularity = {
                    "Vegetable Rice": 0.72,
                    "Chicken Biryani": 0.85,
                    "Idli": 0.68,
                    "Dosa": 0.75,
                    "Chapati": 0.60,
                    "Pongal": 0.65
                }[food]

                meal_factor = 1.0 if meal == "Lunch" else 0.75
                event_factor = 1.10 if event == "Yes" else 1.0

                expected = (
                    attendance * popularity *
                    meal_factor * event_factor
                )

                consumed = max(
                    0, int(rng.normal(expected, 15))
                )

                prepared = max(
                    consumed,
                    int(consumed * rng.uniform(1.04, 1.22))
                )

                leftover = max(0, prepared - consumed)

                records.append({
                    "date": current_date.isoformat(),
                    "meal": meal,
                    "food_item": food,
                    "attendance": attendance,
                    "prepared_quantity": prepared,
                    "consumed_quantity": consumed,
                    "leftover_quantity": leftover,
                    "special_event": event
                })

    return prepare_data(pd.DataFrame(records))


# --------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------

def create_pdf_report(data):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        "FoodWise AI - Food Waste Report",
        styles["Title"]
    ))
    elements.append(Spacer(1, 15))

    total_prepared = data["prepared_quantity"].sum()
    total_consumed = data["consumed_quantity"].sum()
    total_leftover = data["leftover_quantity"].sum()

    waste_percentage = (
        total_leftover / total_prepared * 100
        if total_prepared > 0 else 0
    )

    summary = [
        ["Metric", "Quantity"],
        ["Total Prepared", f"{total_prepared:,.0f}"],
        ["Total Consumed", f"{total_consumed:,.0f}"],
        ["Total Leftover", f"{total_leftover:,.0f}"],
        ["Waste Percentage", f"{waste_percentage:.2f}%"]
    ]

    table = Table(summary, colWidths=[240, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7048A4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 9),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "Food Item Waste Summary",
        styles["Heading2"]
    ))

    grouped = (
        data.groupby("food_item")["leftover_quantity"]
        .sum()
        .reset_index()
        .sort_values("leftover_quantity", ascending=False)
    )

    table_data = [["Food Item", "Leftover Quantity"]]

    for _, row in grouped.iterrows():
        table_data.append([
            str(row["food_item"]),
            f"{row['leftover_quantity']:,.0f}"
        ])

    food_table = Table(table_data, colWidths=[240, 180])
    food_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7048A4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(food_table)
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        "Generated by FoodWise AI",
        styles["Normal"]
    ))

    document.build(elements)
    buffer.seek(0)

    return buffer


# --------------------------------------------------
# SIDEBAR AND DATA SOURCE
# --------------------------------------------------

st.sidebar.markdown("# 🍽️ FoodWise AI")
st.sidebar.caption("Intelligent Food Demand & Waste Management")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Demand Prediction",
        "Waste Monitoring",
        "Data Management",
        "Reports",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Data Source")

data_source = st.sidebar.selectbox(
    "Choose your dataset",
    ["Demo Data", "Saved Records", "Upload CSV"]
)

uploaded_data = None

if data_source == "Demo Data":
    df = generate_demo_data()
    st.sidebar.info(
        "You are viewing synthetic demonstration data, "
        "not real cafeteria records."
    )

elif data_source == "Saved Records":
    raw_data = load_saved_records()

    if raw_data.empty:
        df = pd.DataFrame()
        st.sidebar.warning(
            "No saved records yet. Add records in Data Management."
        )
    else:
        df = prepare_data(raw_data)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload food data CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)
            df = prepare_data(raw_data)

            st.sidebar.success(
                f"Loaded {len(df)} records."
            )

            if st.sidebar.button("Save uploaded records"):
                save_records(df)
                st.sidebar.success("Records saved to database.")

        except Exception as error:
            df = pd.DataFrame()
            st.sidebar.error(str(error))
    else:
        df = pd.DataFrame()
        st.sidebar.info("Upload a CSV file to continue.")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🍽️ FoodWise AI</h1>
    <p>Predict smarter. Prepare better. Waste less.</p>
    <p>An intelligent food demand prediction and waste
    reduction system powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("📊 Food Management Dashboard")

    if df.empty:
        st.warning("No data available. Add or upload records first.")

    else:
        prepared = df["prepared_quantity"].sum()
        consumed = df["consumed_quantity"].sum()
        leftover = df["leftover_quantity"].sum()

        waste_rate = (
            leftover / prepared * 100 if prepared > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Prepared", f"{prepared:,.0f}")
        col2.metric("Total Consumed", f"{consumed:,.0f}")
        col3.metric("Total Leftover", f"{leftover:,.0f}")
        col4.metric("Waste Rate", f"{waste_rate:.1f}%")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("### Food Consumption")
            consumption = (
                df.groupby("food_item")["consumed_quantity"]
                .sum()
                .reset_index()
                .sort_values("consumed_quantity", ascending=False)
            )

            fig = px.bar(
                consumption,
                x="food_item",
                y="consumed_quantity",
                color="consumed_quantity",
                title="Consumption by Food Item",
                color_continuous_scale="Purples"
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### Food Waste")
            waste = (
                df.groupby("food_item")["leftover_quantity"]
                .sum()
                .reset_index()
                .sort_values("leftover_quantity", ascending=False)
            )

            fig = px.pie(
                waste,
                names="food_item",
                values="leftover_quantity",
                title="Leftover Distribution",
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Purples
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recent Food Records")
        st.dataframe(
            df.tail(15),
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------

elif page == "Demand Prediction":

    st.subheader("🤖 AI Food Demand Prediction")

    st.write(
        "Estimate how much of a food item may be consumed "
        "using attendance, meal, day and event information."
    )

    if df.empty:
        st.warning("Add data before using demand prediction.")

    elif len(df) < 30:
        st.warning(
            "At least 30 valid records are recommended "
            "for model training. Add more historical data."
        )

    else:
        model_data = df.copy()
        model_data["date"] = pd.to_datetime(model_data["date"])
        model_data = model_data.sort_values("date")

        split_index = int(len(model_data) * 0.8)

        train_data = model_data.iloc[:split_index]
        test_data = model_data.iloc[split_index:]

        if len(train_data) < 10 or len(test_data) < 2:
            st.warning("Not enough data for a reliable evaluation.")

        else:
            categorical_features = [
                "day", "meal", "food_item", "special_event"
            ]

            numerical_features = ["attendance"]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical_features
                    ),
                    ("numerical", "passthrough", numerical_features)
                ]
            )

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1
                ))
            ])

            X_train = train_data[MODEL_FEATURES]
            y_train = train_data["consumed_quantity"]

            X_test = test_data[MODEL_FEATURES]
            y_test = test_data["consumed_quantity"]

            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, test_predictions)
            rmse = np.sqrt(
                mean_squared_error(y_test, test_predictions)
            )
            r2 = r2_score(y_test, test_predictions)

            st.markdown("### Model Evaluation")

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.2f}")
            m2.metric("RMSE", f"{rmse:.2f}")
            m3.metric("R² Score", f"{r2:.3f}")

            st.caption(
                "Evaluation uses a chronological 80/20 train-test split. "
                "Metrics describe this dataset and are not a guarantee "
                "of future prediction accuracy."
            )

            st.divider()
            st.markdown("### Predict Future Food Demand")

            with st.form("prediction_form"):
                c1, c2 = st.columns(2)

                with c1:
                    prediction_date = st.date_input(
                        "Prediction date",
                        value=date.today() + timedelta(days=1)
                    )

                    selected_meal = st.selectbox(
                        "Meal",
                        sorted(df["meal"].unique())
                    )

                    selected_food = st.selectbox(
                        "Food item",
                        sorted(df["food_item"].unique())
                    )

                with c2:
                    expected_attendance = st.number_input(
                        "Expected attendance",
                        min_value=0,
                        value=300,
                        step=10
                    )

                    special_event = st.selectbox(
                        "Special event?",
                        ["No", "Yes"]
                    )

                submitted = st.form_submit_button(
                    "Predict Food Demand"
                )
