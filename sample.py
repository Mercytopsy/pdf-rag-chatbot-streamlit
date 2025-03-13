import streamlit as st



def main():

    if "chat_active" not in st.session_state:

        st.session_state["chat_active"] = True



    if st.button("Stop Chat"):

        st.session_state["chat_active"] = False



    if st.session_state["chat_active"]:

        user_input = st.chat_input("Enter your message:")

        # ... process user input ...

    else:

        st.chat_input("Chat is currently stopped", disabled=True) 



if __name__ == "__main__":

    main()
