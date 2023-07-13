import streamlit as st

# Define the menu items
menu_items = [
    {"name": "Option A", "icon": "▶️"},
    {"name": "Option B", "icon": "🎙️"},
    {"name": "Option C", "icon": "🔊"},
]

# Function to toggle the menu expansion
def toggle_menu_expansion():
    global menu_expanded
    menu_expanded = not menu_expanded

# Set initial menu expansion state
menu_expanded = False

# Define the Streamlit layout
st.set_page_config(layout="wide")
st.image("Original.png", width=100)
st.title("FirstLanguage ASR Demo")
st.markdown('<style>' + open('./styleTest.css').read() + '</style>', unsafe_allow_html=True)


# Render the menu
with st.markdown('<div class="menu %s">' % ('expanded' if menu_expanded else ''), unsafe_allow_html=True):
    for item in menu_items:
        st.button(item["name"], key=item["name"], on_click=toggle_menu_expansion, help=item["icon"])

# Add some content below the menu
st.write("This is the content below the menu.")
