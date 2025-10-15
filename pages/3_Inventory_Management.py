import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.stop()

st.title("📦 Advanced Inventory Management")
st.markdown("### Comprehensive material tracking and stock control")

# Create sample inventory data
@st.cache_data
def load_inventory():
    materials = ['Steel', 'Cement', 'Conductors', 'Equipment']
    data = []
    
    for i, material in enumerate(materials):
        for j in range(5):  # 5 batches per material
            current_stock = np.random.randint(800, 2500)
            reorder_point = np.random.randint(500, 1000)
            
            # Determine status based on stock vs reorder point
            if current_stock < reorder_point * 0.5:
                status = 'Critical'
            elif current_stock < reorder_point:
                status = 'Low Stock'
            elif current_stock < reorder_point * 1.5:
                status = 'Medium Stock'
            else:
                status = 'In Stock'
            
            data.append({
                'Material_ID': f'M{i}{j:03d}',
                'Material': material,
                'Current_Stock': current_stock,
                'Reorder_Point': reorder_point,
                'Safety_Stock': np.random.randint(200, 500),
                'Unit_Cost': round(np.random.uniform(50, 500), 2),
                'Supplier': f'Supplier {chr(65 + np.random.randint(0, 5))}',
                'Lead_Time': f'{np.random.randint(10, 45)} days',
                'Last_Updated': (datetime.now() - pd.Timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d'),
                'Status': status
            })
    
    return pd.DataFrame(data)

inventory_df = load_inventory()

# Summary Cards
st.markdown("### 📊 Inventory Summary")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Items", len(inventory_df))
with col2:
    in_stock = len(inventory_df[inventory_df['Status'] == 'In Stock'])
    st.metric("In Stock", in_stock, f"{in_stock/len(inventory_df)*100:.1f}%")
with col3:
    medium = len(inventory_df[inventory_df['Status'] == 'Medium Stock'])
    st.metric("Medium Stock", medium)
with col4:
    low_stock = len(inventory_df[inventory_df['Status'] == 'Low Stock'])
    st.metric("Low Stock", low_stock, delta="-2", delta_color="inverse")
with col5:
    critical = len(inventory_df[inventory_df['Status'] == 'Critical'])
    st.metric("Critical", critical, delta="-1", delta_color="inverse")

st.markdown("---")

# Search and Filter Section
st.markdown("### 🔍 Search & Filter")

col1, col2, col3 = st.columns(3)

with col1:
    search_term = st.text_input("🔎 Search Material ID", placeholder="Enter Material ID...")

with col2:
    material_filter = st.multiselect(
        "Material Type", 
        options=inventory_df['Material'].unique(),
        default=inventory_df['Material'].unique()
    )

with col3:
    status_filter = st.multiselect(
        "Status",
        options=['In Stock', 'Medium Stock', 'Low Stock', 'Critical'],
        default=['In Stock', 'Medium Stock', 'Low Stock', 'Critical']  # Show all by default
    )

# Apply filters
filtered_df = inventory_df.copy()

if search_term:
    filtered_df = filtered_df[filtered_df['Material_ID'].str.contains(search_term, case=False)]

# Filter by material type
filtered_df = filtered_df[filtered_df['Material'].isin(material_filter)]

# Filter by status - THIS IS THE KEY FIX
filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]

st.markdown(f"**Showing {len(filtered_df)} of {len(inventory_df)} items**")

st.markdown("---")

# Main Inventory Table
st.markdown("### 📋 Inventory Details")

# Add status indicators with colors
def get_status_icon(status):
    icons = {
        'In Stock': '🟢',
        'Medium Stock': '🟡',
        'Low Stock': '🟠',
        'Critical': '🔴'
    }
    return icons.get(status, '⚪')

# Add colored status column
display_df = filtered_df.copy()
display_df['Status_Icon'] = display_df['Status'].apply(get_status_icon)

# Display table
st.dataframe(
    display_df[['Status_Icon', 'Material_ID', 'Material', 'Current_Stock', 
                'Reorder_Point', 'Safety_Stock', 'Unit_Cost', 'Supplier', 
                'Lead_Time', 'Last_Updated', 'Status']],
    column_config={
        "Status_Icon": st.column_config.TextColumn("", width="small"),
        "Material_ID": "ID",
        "Material": "Material",
        "Current_Stock": st.column_config.NumberColumn("Stock", format="%d units"),
        "Reorder_Point": st.column_config.NumberColumn("Reorder Point", format="%d"),
        "Safety_Stock": st.column_config.NumberColumn("Safety Stock", format="%d"),
        "Unit_Cost": st.column_config.NumberColumn("Unit Cost", format="₹%.2f"),
        "Supplier": "Supplier",
        "Lead_Time": "Lead Time",
        "Last_Updated": "Last Updated",
        "Status": st.column_config.TextColumn("Status", width="medium")
    },
    hide_index=True,
    use_container_width=True
)

# Action Buttons
st.markdown("---")
st.markdown("### ⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("➕ Add New Item", use_container_width=True):
        st.info("Add new item functionality")

with col2:
    if st.button("📥 Import from Excel", use_container_width=True):
        st.info("File upload functionality")

with col3:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📤 Export to CSV",
        data=csv,
        file_name=f"inventory_export_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col4:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Critical Items Alert
st.markdown("---")
st.markdown("### 🚨 Stock Alerts")

# Critical items
critical_items = filtered_df[filtered_df['Status'] == 'Critical']
if len(critical_items) > 0:
    st.error(f"**{len(critical_items)} CRITICAL items require immediate attention!**")
    for _, item in critical_items.head(3).iterrows():
        st.error(f"🔴 **{item['Material']}** (ID: {item['Material_ID']}) - "
                f"Stock: {item['Current_Stock']} units | Reorder: {item['Reorder_Point']} units")

# Low stock items
low_stock_items = filtered_df[filtered_df['Status'] == 'Low Stock']
if len(low_stock_items) > 0:
    with st.expander(f"🟠 View {len(low_stock_items)} Low Stock Items"):
        st.dataframe(
            low_stock_items[['Material_ID', 'Material', 'Current_Stock', 'Reorder_Point', 'Supplier']], 
            hide_index=True, 
            use_container_width=True
        )

# Medium stock items
medium_stock_items = filtered_df[filtered_df['Status'] == 'Medium Stock']
if len(medium_stock_items) > 0:
    with st.expander(f"🟡 View {len(medium_stock_items)} Medium Stock Items"):
        st.dataframe(
            medium_stock_items[['Material_ID', 'Material', 'Current_Stock', 'Reorder_Point', 'Supplier']], 
            hide_index=True, 
            use_container_width=True
        )

# Footer
st.markdown("---")
total_value = (filtered_df['Current_Stock'] * filtered_df['Unit_Cost']).sum()
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
          f"Total inventory value: ₹{total_value/10000000:.2f} Crores")
