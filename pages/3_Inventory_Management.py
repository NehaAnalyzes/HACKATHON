import streamlit as st
import pandas as pd
import numpy as np

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.info("👈 Use the sidebar menu to return to the main page and login")
    st.stop()

st.title("📦 Advanced Inventory Management")
st.markdown("### Comprehensive material tracking and stock control")

# Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Items", "20", "")
with col2:
    st.metric("In Stock", "15", "75%")
with col3:
    st.metric("Low Stock", "3", "-2")
with col4:
    st.metric("Critical", "2", "-1")
with col5:
    st.metric("Total Value", "₹24.5 Cr", "+8.2%")

st.markdown("---")

# Search and Filter
st.markdown("### 🔍 Search & Filter")
col1, col2, col3 = st.columns(3)

with col1:
    search = st.text_input("🔎 Search Material ID", placeholder="Enter ID...")
with col2:
    material_filter = st.multiselect("Material Type", ["Steel", "Cement", "Conductors", "Equipment"],
                                     default=["Steel", "Cement", "Conductors", "Equipment"])
with col3:
    status_filter = st.multiselect("Status", ["In Stock", "Low Stock", "Critical"],
                                   default=["In Stock", "Low Stock", "Critical"])

st.markdown("---")

# Sample Inventory Table
inventory_data = pd.DataFrame({
    'Material_ID': ['M001', 'M002', 'M003', 'M004'],
    'Material': ['Steel', 'Cement', 'Conductors', 'Equipment'],
    'Current_Stock': [1200, 850, 560, 320],
    'Reorder_Point': [500, 400, 300, 200],
    'Unit_Cost': ['₹250', '₹180', '₹450', '₹1200'],
    'Supplier': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D'],
    'Lead_Time': ['20 days', '15 days', '30 days', '25 days'],
    'Status': ['✅ In Stock', '⚠️ Low Stock', '✅ In Stock', '🔴 Critical']
})

st.markdown("### 📋 Inventory Details")
st.dataframe(inventory_data, use_container_width=True, hide_index=True)

# Action Buttons
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("➕ Add New Item", use_container_width=True):
        st.info("Add item functionality")
with col2:
    if st.button("📥 Import Excel", use_container_width=True):
        st.info("Import functionality")
with col3:
    csv = inventory_data.to_csv(index=False)
    st.download_button("📤 Export CSV", csv, "inventory.csv", use_container_width=True)
with col4:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Critical Alerts
st.markdown("---")
st.markdown("### 🚨 Critical Stock Alerts")
st.error("🔴 **Equipment** (M004) - Stock: 320 units | Reorder: 200 units")
st.warning("🟡 **Cement** (M002) - Stock: 850 units | Reorder: 400 units")
