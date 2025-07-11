import streamlit as st
import hashlib

def check_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.get('authenticated', False):
        with st.sidebar:
            st.subheader("Authentication")
            pwd = st.text_input("Password:", type="password")
            if st.button("Login"):
                # For demo purposes only, hashes of the password 'admin'
                if hashlib.sha256(pwd.encode()).hexdigest() == "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918":
                    st.session_state.authenticated = True
                    st.session_state.user = "admin"
                    st.success("Logged in as admin")
                    st.balloons()
                    st.rerun()
                    
                    
                else:
                    st.error("Invalid credentials")
        return False
    return True
