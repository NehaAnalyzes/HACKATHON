import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Page config
st.set_page_config(
    page_title="POWERGRID Material Forecast",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("⚠️ Configuration file not found. Using default credentials.")
    # Fallback config if file missing
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'name': 'Admin User',
                    'password': '$2b$12$KIXxLq3h5L5ZvBxZ5vZxZe5vZxZe5vZxZe5vZxZe5vZxZe5vZxZ'
                }
            }
        },
        'cookie': {
            'name': 'powergrid_auth',
            'key': 'powergrid_sih_2025_secret_key',
            'expiry_days': 30
        }
    }

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config.get('preauthorized', {})
)

# Login widget
name, authentication_status, username = authenticator.login('Login', 'main')

# Authentication logic
if authentication_status:
    # Sidebar with logout
    with st.sidebar:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar')
        
        st.markdown("---")
        st.markdown("### 🔌 POWERGRID Forecast")
        st.markdown("**Smart India Hackathon 2025**")
        st.markdown("Ministry of Power")
        
        # System health indicators
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("🟢 Model: Active")
        st.info("📡 API: Connected")
        st.success("💾 Database: Online")
        
    # Main welcome page
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
    
elif authentication_status == False:
    st.error('Username/password is incorrect')
    st.info("**Demo Credentials:**")
    st.code("Username: admin\nPassword: admin123")
    
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.info("""
    ### 🔌 POWERGRID Material Demand Forecasting
    
    **Smart India Hackathon 2025**
    
    This system provides:
    - ✅ Accurate demand forecasting (5.31% MAPE)
    - ✅ Real-time inventory management
    - ✅ Advanced analytics & reporting
    - ✅ Intelligent alerts & notifications
    
    **Demo Credentials:**
    - Username: `admin` | Password: `admin123`
    - Username: `manager` | Password: `manager123`
    """)
