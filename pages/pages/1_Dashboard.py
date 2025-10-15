import streamlit as st
import pandas as pd
import plotly.express as px

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.info("👈 Use the sidebar menu to return to the main page and login")
    st.stop()

st.title("🏠 Dashboard Overview")
st.markdown("### Real-time Inventory & System Monitoring")

# Key Metrics
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Inventory Value", "₹24.5 Cr", "+8.2%")
with col2:
    st.metric("Active Projects", "1,199", "+12")
with col3:
    st.metric("Low Stock Items", "3", "-2")
with col4:
    st.metric("Pending Orders", "45", "+5")
with col5:
    st.metric("Forecast Accuracy", "94.71%", "+2.1%")

st.markdown("---")

# System Alerts
st.markdown("### 🔔 System Alerts")
col1, col2 = st.columns([2, 1])

with col1:
    st.error("🚨 **CRITICAL**: Material 1 (Steel) below reorder point")
    st.warning("⚠️ **WARNING**: Material 3 lead time increased to 42 days")
    st.info("ℹ️ **INFO**: Quarterly report due in 5 days")
    st.success("✅ **SUCCESS**: Forecast model updated successfully")

with col2:
    st.markdown("### 📑 Recent Reports")
    st.write("📄 Q4 Procurement - Oct 12")
    st.write("📄 Inventory Audit - Oct 10")
    st.write("📄 Cost Analysis - Oct 8")

st.markdown("---")

# Sample data for charts
data = pd.DataFrame({
    'Material': ['Steel', 'Cement', 'Conductors', 'Equipment'],
    'Stock': [1200, 850, 560, 320],
    'Reorder_Point': [500, 400, 300, 200]
})

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📦 Current Stock Levels")
    fig = px.bar(data, x='Material', y='Stock', color='Material', 
                 title='Inventory by Material Type')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🌍 Regional Distribution")
    region_data = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West', 'Central'],
        'Projects': [250, 280, 190, 220, 259]
    })
    fig2 = px.pie(region_data, values='Projects', names='Region',
                  title='Projects by Region')
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Last updated: 2025-10-15 08:58 IST | Status: 🟢 Active")
