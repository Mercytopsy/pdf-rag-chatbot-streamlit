
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
import streamlit as st
import logging
import time

logging.basicConfig(level=logging.INFO)
# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []


def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            cleaned_response = response.replace("<think>", "").replace("</think>", "")
            response_placeholder.write(cleaned_response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

#Streamlit App Setup
def main():
    st.title("Chat with LLMS Models")
    logging.info("App started")

    # Sidebar for model selection
    model = st.sidebar.selectbox("Choose a model", ["llama3", "deepseek-r1:7b", "mistral"]) 
    logging.info(f"Model selected: {model}")

    # Prompt for user input and save to chat history
    if prompt:= st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        #Display the user's query
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate a new response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time() #timing the response generation
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        messages = [ChatMessage(role = msg['role'], content=msg['content']) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time()-start_time #calculate the duration
                        response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                    except Exception as e:
                        st.sesssion_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occured while generating the response.")
                        logging.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()