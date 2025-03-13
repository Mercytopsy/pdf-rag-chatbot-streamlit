import streamlit as st
import time

if "messages" not in st.session_state:
    st.session_state.messages = []
if "disabled" not in st.session_state:
    st.session_state.disabled = False

def disable():
    st.session_state.disabled = True

def save_rating(i):
    st.session_state.messages[i]["feedback"] = st.session_state[f"feedback_{i}"]

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "ai":
            if "feedback" in message:
                st.markdown(":star:"*(message["feedback"]+1))
            else:
                st.feedback("stars", key=f"feedback_{i}", on_change=save_rating, args=[i])

if prompt := st.chat_input("Type one word for demo", disabled=st.session_state.disabled, on_submit=disable):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    with st.chat_message("ai"):
        time.sleep(2)
        st.write(prompt)
    st.session_state.messages.append(
        {
            "role": "ai",
            "content": prompt
        }
    )
    st.session_state.disabled = False
    st.rerun()