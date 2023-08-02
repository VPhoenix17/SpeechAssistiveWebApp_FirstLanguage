import streamlit as st

def main():
    st.title("Grammarly Editor Plugin Integration")
    st.markdown("Use the Grammarly Editor Plugin to enhance your text!")

    # Add the Grammarly Editor plugin script using the provided URL
    st.write("<script src='https://cdn.jsdelivr.net/npm/@grammarly/editor-sdk?clientId=YOUR_CLIENT_ID'></script>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
