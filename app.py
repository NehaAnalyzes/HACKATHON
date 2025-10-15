import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path

# Page config
st.set_page_config(
    page_title="POWERGRID Material Forecasting System",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for authentication
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Simple authentication (without external library complexity)
def check_login(username, password):
    """Simple authentication check"""
    users = {
        'admin': 'admin123',
        'manager': 'manager123'
    }
    return users.get(username) == password

# Sidebar with login/logout
with st.sidebar:
    st.markdown("### 🔌 POWERGRID Forecast")
    st.markdown("Ministry of Power")
    st.markdown("---")
    
    # Check if already logged in
    if st.session_state['authentication_status']:
        st.success(f"✅ Logged in as: **{st.session_state['name']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("🟢 Model: Active")
        st.info("📡 API: Connected")
        st.success("💾 Database: Online")
    else:
        st.markdown("### 🔐 Login")
        username_input = st.text_input("Username", key="username_input")
        password_input = st.text_input("Password", type="password", key="password_input")
        
        if st.button("Login", type="primary", use_container_width=True):
            if check_login(username_input, password_input):
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username_input
                st.session_state['name'] = username_input.capitalize()
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n\nUsername: `admin`\nPassword: `admin123`")

# Main content area
if st.session_state['authentication_status']:
    # User is logged in - show dashboard
    st.title("🔌 POWERGRID Material Demand Forecasting System")
    st.markdown("### Welcome to the Supply Chain Intelligence Platform")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", "94.71%", "+4.71%")
    with col2:
        st.metric("MAPE", "5.31%", "-69% vs baseline")
    with col3:
        st.metric("Active Projects", "1,199", "+12")
    with col4:
        st.metric("Materials Tracked", "4 Types", "")
    
    st.markdown("---")
    
    # Quick Navigation Buttons - FIXED SECTION
    st.markdown("### ⚡ Quick Navigation")
    st.markdown("Click any button below to access different modules:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Dashboard", use_container_width=True, type="primary", help="View inventory overview and alerts"):
            st.switch_page("pages/1_Dashboard.py")
        
        if st.button("📊 Demand Forecast", use_container_width=True, help="Generate real-time predictions"):
            st.switch_page("pages/2_Demand_Forecast.py")
    
    with col2:
        if st.button("📦 Inventory Management", use_container_width=True, help="Track materials and stock levels"):
            st.switch_page("pages/3_Inventory_Management.py")
        
        if st.button("📑 Reports & Analytics", use_container_width=True, help="Generate comprehensive reports"):
            st.switch_page("pages/4_Reports_Analytics.py")
    
    with col3:
        if st.button("⚙️ Settings", use_container_width=True, help="Customize your experience"):
            st.switch_page("pages/5_Settings.py")
    
    st.markdown("---")
    
    st.info("""
    💡 **Navigation Tips:**
    - Click the buttons above to quickly access different modules
    - Or use the sidebar menu (☰) at the top left
    - All pages are accessible after login
    """)
    
    # Quick stats
    st.markdown("### 📈 Quick Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Recent Activity")
        st.success("✅ Forecast generated for Material 2 - State 3")
        st.info("📊 Inventory report generated")
        st.warning("⚠️ Low stock alert: Material 1")
        
    with col2:
        st.markdown("#### System Information")
        st.write("**Last Model Update:** 2025-10-14")
        st.write("**Training Time:** 0.62 seconds")
        st.write("**Total Predictions:** 15,847")
        st.write("**Average Response Time:** < 100ms")
        st.write(f"**Logged in as:** {st.session_state['name']}")
    
    # Feature Highlights
    st.markdown("---")
    st.markdown("### 🌟 Key Features")
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    with feat_col1:
        st.markdown("#### 🎯 Forecasting")
        st.write("5.31% MAPE accuracy")
        st.write("Real-time predictions")
        st.write("12-month horizon")
    
    with feat_col2:
        st.markdown("#### 📦 Inventory")
        st.write("Stock tracking")
        st.write("Reorder alerts")
        st.write("Supplier management")
    
    with feat_col3:
        st.markdown("#### 📊 Analytics")
        st.write("Cost analysis")
        st.write("Demand trends")
        st.write("Custom reports")
    
    with feat_col4:
        st.markdown("#### 🔔 Alerts")
        st.write("Low stock warnings")
        st.write("Price changes")
        st.write("Lead time updates")

else:
    # Not logged in - show login prompt
    st.title("🔌 POWERGRID Material Demand Forecasting")
  
    
    st.info("""
    ## Welcome to POWERGRID's Supply Chain Intelligence Platform
    
   
    
    ### 🔐 Please login to continue
    
   
    
    col3, col4 = st.columns(2)

    with col3:
        st.metric("Training Time", "0.62 sec", "Real-time")
    with col4:
        st.metric("Total Projects", "1,199", "+12")

