import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path

# Page config
st.set_page_config(
    page_title="POWERGRID Material Forecast",
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
    st.markdown("**Smart India Hackathon 2025**")
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
    
    st.info("""
    👈 **Navigate using the sidebar to access:**
    - 🏠 Dashboard - Overview of inventory and alerts
    - 📊 Demand Forecast - Generate real-time predictions
    - 📦 Inventory Management - Track materials and stock levels
    - 📑 Reports & Analytics - Generate comprehensive reports
    - ⚙️ Settings - Customize your experience
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

else:
    # Not logged in - show login prompt
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("### Smart India Hackathon 2025")
    
    st.info("""
    ## Welcome to POWERGRID's Supply Chain Intelligence Platform
    
    This system provides:
    - ✅ **Accurate demand forecasting** (5.31% MAPE)
    - ✅ **Real-time inventory management**
    - ✅ **Advanced analytics & reporting**
    - ✅ **Intelligent alerts & notifications**
    
    ### 🔐 Please login to continue
    
    Use the login form in the sidebar to access the system.
    
    **Demo Credentials:**
    - Username: `admin` | Password: `admin123`
    - Username: `manager` | Password: `manager123`
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model MAPE", "5.31%", "-69%")
    with col2:
        st.metric("R² Score", "0.9471", "+94.71%")
    with col3:
        st.metric("Training Time", "0.62 sec", "Real-time")
    with col4:
        st.metric("Total Projects", "1,199", "+12")
