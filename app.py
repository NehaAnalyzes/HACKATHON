import streamlit as st
from datetime import datetime

# Check authentication
# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.info("👉 Click the link below to go to the login page")
    if st.button("🔐 Go to Login Page", type="primary"):
        st.switch_page("app.py")
    st.stop()


st.title("⚙️ User Settings & Customization")
st.markdown("### Personalize your POWERGRID experience")

# User Profile Section
st.markdown("### 👤 User Profile")

col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://via.placeholder.com/150", caption="Profile Picture")
    if st.button("Upload New Photo", use_container_width=True):
        st.info("Photo upload functionality")

with col2:
    name = st.text_input("Full Name", value=st.session_state.get('name', 'Admin User'))
    email = st.text_input("Email", value="admin@powergrid.com")
    role = st.selectbox("Role", ["Admin", "Manager", "Analyst", "Viewer"])
    department = st.selectbox("Department", ["Procurement", "Operations", "Finance", "IT"])

st.markdown("---")

# Notification Settings
st.markdown("### 🔔 Notification Preferences")

st.markdown("**Low Stock Alerts:**")
col1, col2 = st.columns(2)
with col1:
    low_stock_email = st.checkbox("Email Notifications", value=True)
    low_stock_sms = st.checkbox("SMS Notifications", value=False)
with col2:
    low_stock_threshold = st.slider("Alert Threshold (%)", 0, 50, 20)
    st.caption(f"Alert when stock falls below {low_stock_threshold}% of reorder point")

st.markdown("**Forecast Updates:**")
forecast_updates = st.checkbox("Receive weekly forecast summaries", value=True)
report_notifications = st.checkbox("Monthly report generation alerts", value=True)

st.markdown("**System Alerts:**")
system_critical = st.checkbox("Critical system alerts", value=True)
system_updates = st.checkbox("System update notifications", value=True)

st.markdown("---")

# Application Preferences
st.markdown("### 🎨 Application Preferences")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Theme Settings:**")
    theme = st.selectbox("Color Theme", ["Light", "Dark", "Auto (System)"])
    
    st.markdown("**Display Options:**")
    compact_view = st.checkbox("Compact view mode", value=False)
    show_tooltips = st.checkbox("Show helpful tooltips", value=True)
    
with col2:
    st.markdown("**Language & Region:**")
    language = st.selectbox("Language", ["English", "Hindi", "Tamil", "Telugu", "Bengali"])
    timezone = st.selectbox("Timezone", ["IST (UTC+5:30)", "UTC", "EST", "PST"])
    
    st.markdown("**Data Display:**")
    date_format = st.selectbox("Date Format", ["DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD"])
    currency_format = st.selectbox("Currency", ["₹ INR", "$ USD", "€ EUR"])

st.markdown("---")

# Dashboard Customization
st.markdown("### 📊 Dashboard Customization")

st.markdown("**Default Dashboard View:**")
default_page = st.selectbox("Landing Page", 
                            ["Home", "Dashboard", "Demand Forecast", "Inventory", "Reports"])

st.markdown("**Visible Metrics:**")
col1, col2, col3 = st.columns(3)
with col1:
    show_inventory = st.checkbox("Inventory Status", value=True)
    show_forecasts = st.checkbox("Forecast Accuracy", value=True)
with col2:
    show_alerts = st.checkbox("Active Alerts", value=True)
    show_projects = st.checkbox("Active Projects", value=True)
with col3:
    show_budget = st.checkbox("Budget Utilization", value=True)
    show_reports = st.checkbox("Recent Reports", value=False)

st.markdown("---")

# Data & Privacy
st.markdown("### 🔒 Data & Privacy")

st.markdown("**Data Export:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📥 Export My Data", use_container_width=True):
        st.success("Data export initiated. You'll receive an email when ready.")
with col2:
    if st.button("📊 View Activity Log", use_container_width=True):
        st.info("Activity log: Last login: 2025-10-14 23:40 IST")
with col3:
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")

st.markdown("**Privacy Settings:**")
data_sharing = st.checkbox("Allow anonymous usage analytics", value=True)
st.caption("Helps us improve the platform. No personal data is shared.")

st.markdown("---")

# Advanced Settings
with st.expander("⚙️ Advanced Settings"):
    st.markdown("**API Configuration:**")
    api_enabled = st.checkbox("Enable API access", value=False)
    if api_enabled:
        st.code("API Key: pk_live_abc123xyz789", language="text")
        if st.button("Regenerate API Key"):
            st.warning("Are you sure? This will invalidate the current key.")
    
    st.markdown("**Forecast Model Settings:**")
    model_sensitivity = st.slider("Forecast Sensitivity", 1, 10, 5)
    auto_retrain = st.checkbox("Auto-retrain model weekly", value=True)
    
    st.markdown("**Database Settings:**")
    backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
    data_retention = st.number_input("Data Retention (months)", 6, 60, 24)

st.markdown("---")

# Action Buttons
st.markdown("### 💾 Save Settings")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Save All Changes", type="primary", use_container_width=True):
        st.success("✅ Settings saved successfully!")
        st.balloons()

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.warning("Settings reset to default values")

with col3:
    if st.button("❌ Cancel", use_container_width=True):
        st.info("Changes discarded")

# System Information
st.markdown("---")
st.markdown("### ℹ️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Application Details:**")
    st.write("Version: 1.0.0")
    st.write("Last Updated: 2025-10-14")
    st.write("Model Version: Prophet v1.1.5")
    st.write("Database: PostgreSQL 14.2")

with col2:
    st.markdown("**Support:**")
    st.write("📧 Email: support@powergrid.com")
    st.write("📞 Phone: 1800-XXX-XXXX")
    st.write("🌐 Documentation: docs.powergrid.com")
    st.write("🐛 Report Bug: github.com/powergrid/issues")

# Footer
st.markdown("---")
st.caption(f"Settings last modified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
          f"User: {st.session_state.get('username', 'admin')}")

