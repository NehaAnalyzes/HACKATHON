import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.info("👈 Use the sidebar menu to return to the main page and login")
    st.stop()

st.title("📑 Reports & Analytics")
st.markdown("### Comprehensive reporting and business intelligence")

# Report Generation
st.markdown("### 📊 Generate New Report")

col1, col2, col3 = st.columns(3)

with col1:
    report_type = st.selectbox("Report Type", [
        "Procurement Planning",
        "Cost Analysis",
        "Inventory Status",
        "Supplier Performance"
    ])

with col2:
    date_range = st.selectbox("Time Period", [
        "Last 7 Days", "Last 30 Days", "Last Quarter", "Last Year"
    ])

with col3:
    report_format = st.selectbox("Export Format", ["PDF", "Excel", "CSV"])

if st.button("🔄 Generate Report", type="primary", use_container_width=True):
    st.success(f"✅ {report_type} report generated successfully!")
    
    st.markdown("---")
    st.markdown(f"### 📄 {report_type} Report")
    st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Period:** {date_range}")
    
    # Sample Report Data
    if report_type == "Cost Analysis":
        cost_data = pd.DataFrame({
            'Category': ['Materials', 'Transportation', 'Storage', 'Labor', 'Overhead'],
            'Amount_Cr': [15.5, 2.3, 1.8, 3.2, 1.5],
            'Percentage': [62.8, 9.3, 7.3, 13.0, 6.1]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(cost_data, values='Amount_Cr', names='Category',
                        title='Cost Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(cost_data, use_container_width=True, hide_index=True)
            st.metric("Total Cost", "₹24.7 Crores", "+8.2%")

st.markdown("---")

# Recent Reports
st.markdown("### 📚 Recent Reports")
recent = pd.DataFrame({
    'Report': ['Q4 Procurement', 'Oct Inventory', 'Cost Analysis FY25', 'Supplier Q3'],
    'Date': ['2025-10-12', '2025-10-10', '2025-10-08', '2025-10-05'],
    'Type': ['Procurement', 'Inventory', 'Cost', 'Supplier'],
    'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete']
})
st.dataframe(recent, use_container_width=True, hide_index=True)
