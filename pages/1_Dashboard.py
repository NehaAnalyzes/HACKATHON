import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.stop()

st.title("📊 Dashboard Overview")
st.markdown("### Real-time insights into procurement and inventory status")

st.markdown("---")

# REMOVED: Key Metrics Section

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hybrid_cleaned.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return None

df = load_data()

if df is not None:
    # Material Status Cards
    st.subheader("📦 Material Inventory Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    materials = ['Steel', 'Cement', 'Conductors', 'Equipment']
    colors = ['#FF4B4B', '#FFA500', '#00D4FF', '#00FF00']
    
    for i, (col, material, color) in enumerate(zip([col1, col2, col3, col4], materials, colors)):
        material_data = df[df['Material'] == i]
        total = material_data['Quantity_Procured'].sum()
        
        with col:
            st.markdown(f"""
            <div style='padding: 20px; background-color: {color}22; border-radius: 10px; border-left: 4px solid {color}'>
                <h4 style='margin: 0; color: {color}'>{material}</h4>
                <h2 style='margin: 5px 0'>{total:,.0f}</h2>
                <p style='margin: 0; font-size: 0.9em; opacity: 0.8'>Total Units</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Monthly Procurement Trend
    st.subheader("📈 Monthly Procurement Trends")
    
    monthly = df.groupby(df['Date'].dt.to_period('M'))['Quantity_Procured'].sum().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()
    
    fig = px.line(
        monthly,
        x='Date',
        y='Quantity_Procured',
        title='Total Monthly Procurement',
        labels={'Quantity_Procured': 'Quantity (units)', 'Date': 'Month'}
    )
    fig.update_traces(line_color='#FF4B4B', line_width=3)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Regional Distribution
    st.subheader("🗺️ Regional Procurement Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        region_data = df.groupby('State')['Quantity_Procured'].sum().reset_index()
        region_names = ['North', 'South', 'East', 'West', 'Central']
        region_data['Region'] = region_data['State'].map(lambda x: region_names[x])
        
        fig_pie = px.pie(
            region_data,
            values='Quantity_Procured',
            names='Region',
            title='Quantity by Region'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        material_data = df.groupby('Material')['Quantity_Procured'].sum().reset_index()
        material_data['Material_Name'] = material_data['Material'].map(lambda x: materials[x])
        
        fig_bar = px.bar(
            material_data,
            x='Material_Name',
            y='Quantity_Procured',
            title='Quantity by Material Type',
            color='Material_Name',
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.error("❌ Unable to load data. Please check hybrid_cleaned.csv file.")

st.markdown("---")
st.caption("Dashboard | POWERGRID Material Forecasting System")
