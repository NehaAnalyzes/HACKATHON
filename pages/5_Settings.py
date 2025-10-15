import streamlit as st

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.info("👈 Use the sidebar menu to return to the main page and login")
    st.stop()

st.title("⚙️ User Settings & Customization")
st.markdown("### Personalize your POWERGRID experience")

# User Profile
st.markdown("### 👤 User Profile")
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name", value=st.session_state.get('name', 'Admin'))
    email = st.text_input("Email", value="admin@powergrid.com")

with col2:
    role = st.selectbox("Role", ["Admin", "Manager", "Analyst"])
    department = st.selectbox("Department", ["Procurement", "Operations", "Finance"])

st.markdown("---")

# Notifications
st.markdown("### 🔔 Notification Preferences")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Low Stock Alerts:**")
    low_stock_email = st.checkbox("Email Notifications", value=True)
    low_stock_sms = st.checkbox("SMS Notifications", value=False)

with col2:
    st.markdown("**Threshold Settings:**")
    threshold = st.slider("Alert Threshold (%)", 0, 50, 20)
    st.caption(f"Alert when stock falls below {threshold}%")

st.markdown("---")

# Application Preferences
st.markdown("### 🎨 Application Preferences")
col1, col2 = st.columns(2)

with col1:
    theme = st.selectbox("Color Theme", ["Light", "Dark", "Auto"])
    language = st.selectbox("Language", ["English", "Hindi", "Tamil"])

with col2:
    date_format = st.selectbox("Date Format", ["DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD"])
    timezone = st.selectbox("Timezone", ["IST (UTC+5:30)", "UTC", "EST"])

st.markdown("---")

# Save Button
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Save All Changes", type="primary", use_container_width=True):
        st.success("✅ Settings saved successfully!")
        st.balloons()

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.warning("Settings reset to defaults")

with col3:
    if st.button("❌ Cancel", use_container_width=True):
        st.info("Changes discarded")
