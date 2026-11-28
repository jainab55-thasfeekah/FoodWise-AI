
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import date
from io import BytesIO

st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🍃",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #11111b;
    color: #f4f0ff;
}
[data-testid="stSidebar"] {
    background: #1b1928;
}
h1, h2, h3 {
    color: #c4a5ff !important;
}
.stButton>button {
    background: #8755dc;
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background: #a77af5;
    color: white;
}
[data-testid="stMetric"] {
    background: #211d32;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #44335f;
}
</style>
""", unsafe_allow_html=True)

if "records" not in st.session_state:
    st.session_state.records = pd.DataFrame(
        columns=["Date", "Food", "Prepared", "Consumed", "Wasted"]
    )

st.title("🍃 FoodWise AI")
st.caption("AI-Powered Food Waste Prediction & Prevention System")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Module",
    [
        "Dashboard",
        "AI Demand Prediction",
        "Food Waste Tracker",
        "Dataset & Model",
        "Smart Recommendations",
        "Reports"
    ]
)

def sample_data():
    rng = np.random.default_rng(42)
    days = 180
    data = pd.DataFrame({
        "Day": np.arange(days),
        "DayOfWeek": np.arange(days) % 7,
        "Temperature": rng.uniform(20, 38, days),
        "SpecialEvent": rng.integers(0, 2, days),
        "Attendance": rng.integers(80, 301, days)
    })
    data["Demand"] = (
        data["Attendance"] * 0.82
        + data["SpecialEvent"] * 18
        - (data["Temperature"] - 28) * 0.5
        + rng.normal(0, 9, days)
    ).clip(20).round()
    return data

if "dataset" not in st.session_state:
    st.session_state.dataset = sample_data()

def train_model(df):
    features = [
        "DayOfWeek", "Temperature",
        "SpecialEvent", "Attendance"
    ]
    X = df[features]
    y = df["Demand"]
    if len(df) < 10:
        return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    score = r2_score(y_test, predictions)
    return model, features, mae, score

dataset = st.session_state.dataset
model, features, mae, score = train_model(dataset)

if page == "Dashboard":
    st.header("Your Food Sustainability Overview")
    records = st.session_state.records

    total_prepared = records["Prepared"].sum()
    total_consumed = records["Consumed"].sum()
    total_wasted = records["Wasted"].sum()
    waste_rate = (
        total_wasted / total_prepared * 100
        if total_prepared > 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Food Prepared", f"{total_prepared:.0f}")
    c2.metric("Food Consumed", f"{total_consumed:.0f}")
    c3.metric("Food Wasted", f"{total_wasted:.0f}")
    c4.metric("Waste Rate", f"{waste_rate:.1f}%")

    st.subheader("Food Waste Analytics")
    if not records.empty:
        fig = px.bar(
            records, x="Date",
            y=["Consumed", "Wasted"],
            barmode="group",
            title="Daily Food Consumption vs Waste",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add food records in Food Waste Tracker to view analytics.")

    st.subheader("AI Model Status")
    st.success("Food demand prediction model trained successfully.")
    st.write(f"Model MAE: {mae:.2f} food portions")
    st.write(f"Model R² score: {score:.3f}")

elif page == "AI Demand Prediction":
    st.header("🤖 AI Food Demand Prediction")
    st.write("Enter the expected cafeteria conditions.")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.number_input(
            "Expected Student Attendance",
            min_value=1, max_value=2000, value=200
        )
        temperature = st.slider(
            "Temperature (°C)", 10, 45, 28
        )
    with col2:
        weekday = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"]
        )
        event = st.selectbox(
            "Special Event?", ["No", "Yes"]
        )

    if st.button("Predict Food Demand"):
        prediction = model.predict(pd.DataFrame([{
            "DayOfWeek": [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ].index(weekday),
            "Temperature": temperature,
            "SpecialEvent": 1 if event == "Yes" else 0,
            "Attendance": attendance
        }]))[0]

        recommended = int(round(prediction))
        st.success(f"Predicted demand: {recommended} portions")
        st.metric("Recommended preparation quantity", recommended)
        st.info(
            "Prepare approximately the predicted quantity. "
            "Adjust for actual attendance and serving size."
        )

elif page == "Food Waste Tracker":
    st.header("📊 Food Waste Tracker")

    with st.form("food_form"):
        record_date = st.date_input("Date", date.today())
        food = st.text_input("Food Item", placeholder="Example: Rice")
        prepared = st.number_input(
            "Quantity Prepared", min_value=0, value=100
        )
        consumed = st.number_input(
            "Quantity Consumed", min_value=0, value=80
        )
        submitted = st.form_submit_button("Save Food Record")

        if submitted:
            if not food.strip():
                st.error("Please enter a food item.")
            elif consumed > prepared:
                st.error("Consumed quantity cannot exceed prepared quantity.")
            else:
                wasted = prepared - consumed
                new_record = pd.DataFrame([{
                    "Date": str(record_date),
                    "Food": food.strip(),
                    "Prepared": prepared,
                    "Consumed": consumed,
                    "Wasted": wasted
                }])
                st.session_state.records = pd.concat(
                    [st.session_state.records, new_record],
                    ignore_index=True
                )
                st.success("Food record saved successfully!")

    st.subheader("Recorded Food Data")
    st.dataframe(
        st.session_state.records,
        use_container_width=True
    )

    if st.button("Clear All Records"):
        st.session_state.records = pd.DataFrame(
            columns=["Date", "Food", "Prepared", "Consumed", "Wasted"]
        )
        st.rerun()

elif page == "Dataset & Model":
    st.header("📂 Dataset & Machine Learning")

    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"]
    )

    if uploaded is not None:
        try:
            custom = pd.read_csv(uploaded)
            required = [
                "DayOfWeek", "Temperature",
                "SpecialEvent", "Attendance", "Demand"
            ]
            if all(c in custom.columns for c in required):
                if len(custom) >= 10 and custom[required].notna().all().all():
                    st.session_state.dataset = custom[required].copy()
                    st.success("Dataset loaded successfully.")
                    st.rerun()
                else:
                    st.error("Dataset needs at least 10 complete rows.")
            else:
                st.error("Missing required columns: " + ", ".join(required))
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")

    st.subheader("Current Dataset")
    st.dataframe(dataset.head(20), use_container_width=True)
    st.write(f"Total records: {len(dataset)}")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"R² Score: {score:.3f}")

    st.download_button(
        "Download Sample Dataset",
        dataset.to_csv(index=False),
        "foodwise_sample_dataset.csv",
        "text/csv"
    )

    if st.button("Restore Sample Dataset"):
        st.session_state.dataset = sample_data()
        st.rerun()

elif page == "Smart Recommendations":
    st.header("💡 Smart Food Waste Recommendations")

    records = st.session_state.records

    if records.empty:
        st.info("Add food records to receive data-based recommendations.")
    else:
        total = records["Prepared"].sum()
        waste = records["Wasted"].sum()
        rate = waste / total * 100 if total else 0

        st.metric("Overall Waste Percentage", f"{rate:.2f}%")

        if rate > 20:
            st.warning("High food wastage detected. Reduce preparation quantities and review attendance.")
        elif rate > 10:
            st.info("Moderate food wastage. Monitor daily demand.")
        else:
            st.success("Good food utilization. Continue monitoring.")

        food_summary = records.groupby("Food")[
            ["Prepared", "Consumed", "Wasted"]
        ].sum().sort_values("Wasted", ascending=False)

        st.subheader("Food Items with Highest Waste")
        st.dataframe(food_summary, use_container_width=True)

        highest = food_summary.index[0]
        st.write(f"Review preparation quantities for **{highest}**.")

        st.write(
            "Suggestions: Use historical demand, "
            "encourage portion control, and safely redistribute "
            "eligible surplus food through approved channels."
        )

elif page == "Reports":
    st.header("📄 FoodWise AI Reports")

    records = st.session_state.records

    if records.empty:
        st.info("Add food records before generating reports.")
    else:
        st.subheader("Food Waste Report")
        st.dataframe(records, use_container_width=True)

        csv = records.to_csv(index=False)
        st.download_button(
            "Download CSV Report",
            csv,
            "FoodWise_AI_Report.csv",
            "text/csv"
        )

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            records.to_excel(
                writer, index=False, sheet_name="Food Waste"
            )

        st.download_button(
            "Download Excel Report",
            buffer.getvalue(),
            "FoodWise_AI_Report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.sidebar.markdown("---")
st.sidebar.caption("FoodWise AI | Sustainable Food Management")
st.sidebar.caption("AI Innovation Prototype")
