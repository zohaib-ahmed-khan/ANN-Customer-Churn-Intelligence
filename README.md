# ANN-Customer-Churn-Intelligence & Risk Simulator

An end-to-end customer retention platform powered by an Artificial Neural Network (ANN). Built to help businesses predict customer churn, run interactive "What-If" risk scenarios, process large batch files automatically, and track revenue at risk.

---

## 🌟 Why This Project Stands Out (Beyond Simple Prediction)

Standard churn models only give a static percentage score. This application functions as an **advanced, end-to-end Decision Support System**:

* 📁 **Bulk CSV Processing & High-Risk Export:**
  Skip checking records one by one—upload your entire customer database at once. The system evaluates the whole batch, calculates overall financial risk, and provides two ready-to-use CSV files:
  1. **Full Prediction CSV:** The complete dataset with churn probabilities, active thresholds, and risk bands.
  2. **High-Risk Target CSV:** A dedicated list containing only critical-risk customers for immediate retention outreach.

* 🧪 **Interactive What-If Scenario Simulator:**
  Exclusively available during single-customer analysis. Instead of just viewing a static risk score, tweak actionable customer factors (*Is Active Member*, *Number of Products*, *Balance*) in real time to immediately compare **Original Churn Risk vs. Renewed/Simulated Risk**.

* 💳 **At-Risk Revenue Metrics (Capital at Risk):**
  Quantifies churn in actual monetary value ($) by dynamically adding up account balances linked to high-risk customers across single and bulk modes.

* 📄 **Automated PDF Risk Reports:**
  Generate and download an official executive PDF report featuring customer profiles, risk metrics, and custom retention strategies with a single click.

---

## 🔥 Core Features

### 1. 👤 Single Customer Risk Analysis & Simulation
* **Real-Time ANN Scoring:** Dynamic risk classification across 3 bands (**🟢 Low Risk**, **🟡 Medium Risk**, **🔴 High Risk**).
* **3D Visualizations & Gauges:** Features interactive Plotly 3D spatial vector plots and neural gauge meters.
* **What-If Simulator Engine:** Live comparative delta metric showing **Original Risk vs. Simulated Risk** without page resets.
* **Executive PDF Generation:** Instantly exports a printable retention roadmap for account managers.

### 2. 📁 Bulk CSV Batch Processing
* **Scalable Batch Auditing:** Processes thousands of customer records concurrently using optimized model vectorization.
* **Automated Capital-at-Risk Aggregation:** Instantly calculates the total monetary value at risk across the uploaded portfolio.
* **Dual CSV Output:** One-click exports for full dataset results and critical-risk-only lists.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend Framework:** Streamlit (Custom Glassmorphism CSS & Vanilla Tilt 3D interactions)
* **Machine Learning / Deep Learning:** TensorFlow / Keras (Artificial Neural Network)
* **Data Processing & Analytics:** Pandas, NumPy, Scikit-Learn
* **3D Visualizations:** Plotly Graph Objects
* **Document Generation:** FPDF

---

## 🚀 Quickstart & Deployment

### 🌐 Live Demo
Access the live interactive application here:
👉 **[Live Streamlit App](https://your-app-name.streamlit.app)** *(Replace with your actual Streamlit App URL)*

---

### 💻 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/ann-customer-churn-intelligence.git](https://github.com/your-username/ann-customer-churn-intelligence.git)
   cd ann-customer-churn-intelligence
