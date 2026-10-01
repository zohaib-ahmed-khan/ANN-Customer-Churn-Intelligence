import io
import pickle
from pathlib import Path 
import numpy as np 
import pandas as pd 
import streamlit as st 
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras 
from fpdf import FPDF

# Setup dynamic path to script directory
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# MODEL CACHING & LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_ann_model():
    # Dynamic path binding with exact filename
    return tf.keras.models.load_model(BASE_DIR / 'churn_model.keras')

model = load_ann_model()

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Churn Intelligence 3D",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Master Glassmorphism & Cyberpunk CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

    :root {
        --bg-main: #05070f;
        --card-glass: rgba(18, 25, 41, 0.65);
        --card-border: rgba(255, 255, 255, 0.12);
        --card-border-glow: rgba(0, 242, 254, 0.6);
        --neon-cyan: #00f2fe;
        --neon-purple: #9d4edd;
        --neon-pink: #ff2a5f;
        --text-pure: #ffffff;
        --text-sub: #cbd5e1;
    }

    .stApp, section[data-testid="stSidebar"] {
        background: var(--bg-main) !important;
        background-image: 
            radial-gradient(at 10% 10%, rgba(127, 0, 255, 0.25) 0px, transparent 50%),
            radial-gradient(at 90% 10%, rgba(0, 242, 254, 0.2) 0px, transparent 50%),
            radial-gradient(at 50% 90%, rgba(255, 42, 95, 0.15) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-pure) !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(25px) !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .hero-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.8rem, 5.5vw, 4.8rem) !important;
        font-weight: 800;
        text-align: center;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 10%, #00f2fe 45%, #9d4edd 75%, #ff2a5f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 15px 30px rgba(0, 242, 254, 0.35));
        margin-top: 10px;
        margin-bottom: 0px;
        line-height: 1.05;
    }

    .hero-subheading {
        text-align: center;
        font-size: 1.15rem;
        color: var(--text-sub) !important;
        letter-spacing: 0.12em;
        margin-bottom: 2rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-pure) !important;
        font-size: 1.35rem !important;
    }

    .card-3d {
        position: relative;
        background: var(--card-glass);
        backdrop-filter: blur(25px) saturate(180%);
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid var(--card-border);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .card-3d:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: var(--card-border-glow);
        box-shadow: 0 30px 60px rgba(0, 242, 254, 0.25);
    }

    .card-title {
        color: var(--text-sub) !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin-bottom: 10px;
    }

    .card-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    div[data-baseweb="input"], div[data-baseweb="select"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: var(--text-pure) !important;
    }

    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #3b82f6 50%, #9d4edd 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        padding: 16px 32px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase;
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.4) !important;
        transition: all 0.4s ease !important;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-4px) scale(1.01) !important;
        box-shadow: 0 18px 45px rgba(157, 78, 221, 0.6) !important;
    }

    div[data-testid="stFileUploader"] {
        background: var(--card-glass);
        border: 2px dashed rgba(0, 242, 254, 0.4);
        border-radius: 20px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3D Tilt Script Integration
st.components.v1.html("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanilla-tilt/1.8.1/vanilla-tilt.min.js"></script>
    <script>
        window.parent.document.querySelectorAll('.card-3d').forEach(el => {
            VanillaTilt.init(el, { max: 12, speed: 400, glare: true, "max-glare": 0.15 });
        });
    </script>
""", height=0)

# ---------------------------------------------------------
# 3. PDF Generator Helper
# ---------------------------------------------------------
def generate_pdf(customer_dict, prob, risk_band, strategy):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(190, 10, txt="CUSTOMER CHURN AI RISK REPORT", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(95, 8, txt=f"Churn Probability: {prob:.1%}", border=1)
    pdf.cell(95, 8, txt=f"Risk Classification: {risk_band}", border=1, ln=True)
    pdf.ln(8)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, txt="Customer Profile Snapshot", ln=True)
    pdf.set_font("Arial", size=10)
    for k, v in customer_dict.items():
        pdf.cell(95, 7, txt=f"{k}:", border=1)
        pdf.cell(95, 7, txt=f"{v}", border=1, ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, txt="Actionable Retention Protocol", ln=True)
    pdf.set_font("Arial", size=10)
    for step in strategy:
        pdf.multi_cell(190, 6, txt=f"- {step}")
        
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------
# 4. Load Artifacts
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(BASE_DIR / "churn_model.keras")

    with open(BASE_DIR / 'gender_encoder.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)

    with open(BASE_DIR / 'geography_encoder.pkl', 'rb') as f:
        geography_encoder = pickle.load(f)

    with open(BASE_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(BASE_DIR / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, gender_encoder, geography_encoder, scaler, feature_names

model, gender_encoder, geography_encoder, scaler, feature_names = load_artifacts()

def prepare_batch_dataframe(df_input):
    df = df_input.copy()
    df['Gender'] = gender_encoder.transform(df[['Gender']]).ravel()
    geo_array = geography_encoder.transform(df[['Geography']])
    geo_cols = geography_encoder.get_feature_names_out(['Geography'])
    geo_df = pd.DataFrame(geo_array, columns=geo_cols, index=df.index)
    df = pd.concat([df.drop(columns='Geography'), geo_df], axis=1)
    df = df.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(df)

# ---------------------------------------------------------
# 5. Header & Sidebar
# ---------------------------------------------------------
st.markdown("<h1 class='hero-heading'>⚡ CUSTOMER CHURN AI 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subheading'>🔮 Next-Gen Deep Learning Predictive Analytics 📊</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Decision Threshold", 0.20, 0.80, 0.50, 0.05)
    st.markdown("""
    ---
    ### 🎨 Risk Scale Matrix
    * **🟢 Safe Level:** `< 30%` (Neon Mint)
    * **🟡 Warning Level:** `30% - 60%` (Vivid Amber)
    * **🔴 Critical Alert:** `>= 60%` (Crimson Red)
    """)

# ---------------------------------------------------------
# 6. Prediction Mode Selector
# ---------------------------------------------------------
mode = st.radio("Choose Prediction Mode:", ["👤 Single Customer Analysis", "📁 Bulk CSV Batch Prediction"], horizontal=True)

st.write("")

if mode == "👤 Single Customer Analysis":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 **Customer Profile**")
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)

    with col2:
        st.markdown("### 💳 **Financial Profile**")
        balance = st.number_input("Balance ($)", 0.0, 3000000.0, 60000.0, step=1000.0)
        num_products = st.number_input("Number of Products", 1, 4, 2)
        has_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 50000.0, step=1000.0)

    customer = {
        'CreditScore': credit_score,
        'Geography': geography, 
        'Gender': gender,   
        'Age': age,    
        'Tenure': tenure,   
        'Balance': balance, 
        'NumOfProducts': num_products,  
        'HasCrCard': has_card,  
        'IsActiveMember': active,   
        'EstimatedSalary': salary
    }

    # Button click par Session State update karna
    if st.button("RUN NEURAL NETWORK PREDICTION", type="primary", use_container_width=True):
        x = prepare_batch_dataframe(pd.DataFrame([customer]))
        st.session_state['has_predicted'] = True
        st.session_state['base_customer'] = customer
        st.session_state['base_prob'] = float(model.predict(x, verbose=0)[0][0])

    # Agar prediction run ho chuki ho toh state maintain rahegi
    if st.session_state.get('has_predicted', False):
        customer = st.session_state['base_customer']
        probability = st.session_state['base_prob']

        if probability >= 0.60:
            band = "HIGH RISK"
            color_code = "#ff2a5f"
            alert_title = "🚨 CRITICAL CHURN DANGER ALERT"
            alert_msg = "Customer displays extreme churn signals. Immediate retention protocol recommended!"
            strategy = [
                "Assign Dedicated Account Manager immediately.",
                "Offer a customized 20% annual discount or loyalty cashback.",
                "Schedule an executive feedback call within 24 hours."
            ]
        elif probability >= 0.30:
            band = "MEDIUM RISK"
            color_code = "#ffb703"
            alert_title = "⚠️ MODERATE WARNING"
            alert_msg = "Customer behavior shows moderate risk. Consider engagement offers."
            strategy = [
                "Send an automated customer satisfaction survey.",
                "Provide promotional access to premium features for 30 days."
            ]
        else:
            band = "LOW RISK"
            color_code = "#00f5d4"
            alert_title = "🟢 SAFE STATUS DETECTED"
            alert_msg = "Customer loyalty is strong. High retention probability."
            strategy = [
                "Maintain current relationship management pipeline.",
                "Consider cross-selling complementary products."
            ]

        st.write("---")
        
        if probability >= 0.60:
            st.error(f"### {alert_title}\n{alert_msg}")
        elif probability >= 0.30:
            st.warning(f"### {alert_title}\n{alert_msg}")
        else:
            st.success(f"### {alert_title}\n{alert_msg}")

        st.write("")

        # 3D Metric Cards + At Risk Revenue
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title">Churn Probability</div>
                <div class="card-value" style="color: {color_code}; text-shadow: 0 0 15px {color_code}aa;">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title">Risk Classification</div>
                <div class="card-value" style="color: {color_code}; text-shadow: 0 0 15px {color_code}aa;">{band}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title">Active Threshold</div>
                <div class="card-value" style="color: #00f2fe; text-shadow: 0 0 15px #00f2feaa;">{threshold:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            at_risk_amount = customer['Balance'] if probability >= threshold else 0.0
            st.markdown(f"""
            <div class="card-3d">
                <div class="card-title">Capital At Risk</div>
                <div class="card-value" style="color: {'#ff2a5f' if at_risk_amount > 0 else '#00f5d4'}; text-shadow: 0 0 15px #ff2a5faa;">${at_risk_amount:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>AI NEURAL GAUGE (%)</b>", 'font': {'size': 18, 'color': "white"}},
                number={'suffix': "%", 'font': {'size': 32, 'color': "white", 'family': "Space Grotesk"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white"},
                    'bar': {'color': color_code},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 245, 212, 0.15)'},
                        {'range': [30, 60], 'color': 'rgba(255, 183, 3, 0.15)'},
                        {'range': [60, 100], 'color': 'rgba(255, 42, 95, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 4},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=340)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col2:
            np.random.seed(42)
            fig_3d = go.Figure()
            fig_3d.add_trace(go.Scatter3d(
                x=np.random.randint(18, 80, 60), y=np.random.randint(0, 200000, 60), z=np.random.randint(400, 850, 60),
                mode='markers', marker=dict(size=4, color='#64748b', opacity=0.4), name='Dataset Pool'
            ))
            fig_3d.add_trace(go.Scatter3d(
                x=[customer['Age']], y=[customer['Balance']], z=[customer['CreditScore']],
                mode='markers+text', marker=dict(size=14, color=color_code, symbol='diamond', line=dict(color='white', width=2)),
                text=['Target Profile'], textposition="top center", name='Active Input'
            ))
            fig_3d.update_layout(
                title="<b>3D SPATIAL DATA VECTOR</b>",
                scene=dict(
                    xaxis_title='Age', yaxis_title='Balance', zaxis_title='Credit Score',
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155")
                ),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), margin=dict(l=0, r=0, b=0, t=30), height=340
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        # ---------------------------------------------------------
        # WHAT-IF SCENARIO SIMULATOR (STABLE STATE)
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧪 **What-If Scenario Simulator**")
        st.caption("Changing the slider or selectbox will instantly update the live prediction calculation without resetting the page..")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_active = st.selectbox("Set Active Member Status", [1, 0], index=0 if customer['IsActiveMember'] == 1 else 1, format_func=lambda x: "Active" if x == 1 else "Inactive", key="sim_act")
        with sim_col2:
            sim_products = st.slider("Adjust Number of Products", 1, 4, int(customer['NumOfProducts']), key="sim_prod")
        with sim_col3:
            sim_balance = st.number_input("Simulate New Balance ($)", 0.0, 3000000.0, float(customer['Balance']), step=5000.0, key="sim_bal")

        sim_customer = customer.copy()
        sim_customer['IsActiveMember'] = sim_active
        sim_customer['NumOfProducts'] = sim_products
        sim_customer['Balance'] = sim_balance

        sim_x = prepare_batch_dataframe(pd.DataFrame([sim_customer]))
        sim_probability = float(model.predict(sim_x, verbose=0)[0][0])
        prob_diff = sim_probability - probability

        s1, s2 = st.columns(2)
        with s1:
            st.metric("Original Churn Risk", f"{probability:.1%}")
        with s2:
            st.metric("Simulated New Churn Risk", f"{sim_probability:.1%}", delta=f"{prob_diff:.1%}", delta_color="inverse")

        st.write("")

        pdf_bytes = generate_pdf(customer, probability, band, strategy)
        st.download_button(
            label="📄 DOWNLOAD PDF RISK REPORT",
            data=pdf_bytes,
            file_name=f"Churn_Report_{customer['CreditScore']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ---------------------------------------------------------
# 7. BULK CSV BATCH PREDICTION MODULE (WITH REVENUE METRICS)
# ---------------------------------------------------------
else:
    st.markdown("### 📁 **Bulk Customer Batch Processing**")
    st.info("💡 Upload a CSV file containing customer data. Required columns: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.")

    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("### 📊 **Dataset Preview**", batch_df.head())

            if st.button("⚡ EXECUTE BATCH NEURAL PREDICTION", type="primary", use_container_width=True):
                with st.spinner("Processing deep learning model over batch data..."):
                    processed_x = prepare_batch_dataframe(batch_df)
                    probs = model.predict(processed_x, verbose=0).ravel()

                    results_df = batch_df.copy()
                    results_df['Churn_Probability_Num'] = probs
                    results_df['Churn_Probability'] = (probs * 100).round(2).astype(str) + "%"
                    results_df['Risk_Level'] = np.where(probs >= 0.60, 'HIGH RISK', np.where(probs >= 0.30, 'MEDIUM RISK', 'LOW RISK'))
                    results_df['Action_Required'] = np.where(probs >= threshold, 'RETENTION PROTOCOL TRIGGERED', 'SAFE')

                high_risk_count = int(np.sum(probs >= 0.60))
                med_risk_count = int(np.sum((probs >= 0.30) & (probs < 0.60)))
                low_risk_count = int(np.sum(probs < 0.30))

                # At-Risk Revenue Metric Calculation
                at_risk_df = results_df[results_df['Churn_Probability_Num'] >= threshold]
                total_at_risk_revenue = at_risk_df['Balance'].sum() if 'Balance' in at_risk_df.columns else 0.0

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.markdown(f'<div class="card-3d"><div class="card-title">Total Customers</div><div class="card-value" style="color: #00f2fe;">{len(batch_df)}</div></div>', unsafe_allow_html=True)
                with b2:
                    st.markdown(f'<div class="card-3d"><div class="card-title">High Risk Alert</div><div class="card-value" style="color: #ff2a5f;">{high_risk_count}</div></div>', unsafe_allow_html=True)
                with b3:
                    st.markdown(f'<div class="card-3d"><div class="card-title">Medium Risk</div><div class="card-value" style="color: #ffb703;">{med_risk_count}</div></div>', unsafe_allow_html=True)
                with b4:
                    st.markdown(f'<div class="card-3d"><div class="card-title">Total Capital At Risk</div><div class="card-value" style="color: #ff2a5f; font-size: 1.8rem;">${total_at_risk_revenue:,.0f}</div></div>', unsafe_allow_html=True)

                st.write("")
                st.markdown("### 📋 **Batch Prediction Results**")
                display_df = results_df.drop(columns=['Churn_Probability_Num'])
                st.dataframe(display_df, use_container_width=True)

                high_risk_df = display_df[results_df['Churn_Probability_Num'] >= 0.60]
                csv_buffer = display_df.to_csv(index=False).encode('utf-8')
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.download_button("📥 DOWNLOAD FULL BATCH RESULTS (CSV)", data=csv_buffer, file_name="Batch_Churn_Predictions_Full.csv", mime="text/csv", use_container_width=True)
                with c_col2:
                    high_risk_csv = high_risk_df.to_csv(index=False).encode('utf-8')
                    st.download_button("🚨 DOWNLOAD HIGH RISK LIST ONLY (CSV)", data=high_risk_csv, file_name="High_Risk_Churn_Customers.csv", mime="text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")