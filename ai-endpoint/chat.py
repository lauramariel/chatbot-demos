"""
GPT-in-a-Box Streamlit App
This module defines a Streamlit app for interacting with different Large Language models.
"""

import os
import json
import sys
import requests
import streamlit as st
import time

# Add supported models to the list
AVAILABLE_MODELS = ["mistral"]
# AVAILABLE_MODELS = ["llama2-7b", "mpt-7b" , "falcon-7b"]
ASSISTANT_SVG = "assistant.svg"
USER_SVG = "user.svg"
LOGO_SVG = "nutanix.svg"

LLM_MODE = "chat"
LLM_HISTORY = "off"

if not os.path.exists(ASSISTANT_SVG):
    ASSISTANT_AVATAR = None
else:
    ASSISTANT_AVATAR = ASSISTANT_SVG

if not os.path.exists(USER_SVG):
    USER_AVATAR = None
else:
    USER_AVATAR = USER_SVG

# App title
st.title("Welcome to .NEXT Barcelona :flag-es:")
st.subtitle("Connected to: 10.117.60.85")

def clear_chat_history():
    """
    Clears the chat history by resetting the session state messages.
    """
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


with st.sidebar:
    if os.path.exists(LOGO_SVG):
        _, col2, _, _ = st.columns(4)
        with col2:
            st.image(LOGO_SVG, width=150)

    st.title("GPT-in-a-Box")
    st.markdown(
        "GPT-in-a-Box is a turnkey AI solution for organizations wanting to implement GPT"
        "capabilities while maintaining control of their data and applications. Read the "
        "[blog]"
        "(http://nutanix.com/blog/gpt-in-a-box-2-is-here)"
    )

    st.subheader("Models")
    selected_model = st.sidebar.selectbox(
        "Choose a model", AVAILABLE_MODELS, key="selected_model"
    )
    if selected_model == "mistral":
        LLM = "mistral"
        st.markdown(
            "Mistral"
        )
    else:
        sys.exit()

    if "model" in st.session_state and st.session_state["model"] != LLM:
        clear_chat_history()

    st.session_state["model"] = LLM

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


def add_message(chatmessage):
    """
    Adds a message to the chat history.
    Parameters:
    - chatmessage (dict): A dictionary containing role ("assistant" or "user")
                      and content of the message.
    """

    if chatmessage["role"] == "assistant":
        avatar = ASSISTANT_AVATAR
    else:
        avatar = USER_AVATAR
    if LLM_MODE == "code":
        with st.chat_message(chatmessage["role"], avatar=avatar):
            st.code(chatmessage["content"], language="python")
    else:
        with st.chat_message(chatmessage["role"], avatar=avatar):
            st.write(chatmessage["content"])


# Display or clear chat messages
for message in st.session_state.messages:
    add_message(message)

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_response(input_text):
    """
    Generates a response from the LLM based on the given prompt.

    Parameters:
    - prompt_input (str): The input prompt for generating a response.

    Returns:
    - str: The generated response.

    """
    input_prompt = get_json_format_prompt(input_text)
    url = f"https://10.117.60.85:31334/api/v1/chat/completions"
    headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": "Bearer 73aa7447-c1e0-4515-a6c5-d94eff0e47f3", "accept": "application/json"}
    body = {
      "model": "mistral",
      "messages": [
        {
          "role": "user",
          "content": input_text
        }
      ],
      "max_tokens": 256,
      "stream": False
}
    try:
        print(f"Sending {json.dumps(body)}")
        start = time.perf_counter()
        response = requests.post(url, json=body, timeout=600, headers=headers, verify=False)
        request_time = time.perf_counter() - start
        print(request_time)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error in requests: ", url, "", e)
        return ""
    output_dict = json.loads(response.text)
    output = output_dict["choices"][0]["message"]["content"]  
    return output


def generate_chat_response(input_prompt):
    """
    Generates a chat-based response by including the chat history in the input prompt.

    Parameters:
    - prompt_input (str): The user-provided prompt.

    Returns:
    - str: The generated chat-based response.

    """
    # Used [INST] and <<SYS>> tags in the input prompts for LLAMA 2 models.
    # These are tags used to indicate different types of input within the conversation.
    # "INST" stands for "instruction" and used to provide user queries to the model.
    # "<<SYS>>" signifies system-related instructions and used to prime the
    # model with context, instructions, or other information relevant to the use case.

    string_dialogue = (
        "[INST] <<SYS>> You are a helpful assistant. "
        " You answer the question asked by 'User' once"
        " as 'Assistant'. <</SYS>>[/INST]" + "\n\n"
    )

    for dict_message in st.session_state.messages[:-1]:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "[/INST]" + "\n\n"
        else:
            string_dialogue += (
                "Assistant: " + dict_message["content"] + " [INST]" + "\n\n"
            )
    string_dialogue += "User: " + f"{input_prompt}" + "\n\n"
    input_text = f"{string_dialogue}" + "\n\n" + "Assistant: [/INST]"
    output = generate_response(input_text)
    # Generation failed
    if len(output) <= len(input_text):
        return ""
    response = output[len(input_text) :]
    return response


# User-provided prompt
if prompt := st.chat_input("Ask your query"):
    message = {"role": "user", "content": prompt}
    st.session_state.messages.append(message)
    add_message(message)


# def get_json_format_prompt(prompt_input):
#     """
#     Converts the input prompt into the JSON format expected by the LLM.

#     Parameters:
#     - prompt_input (str): The input prompt.

#     Returns:
#     - dict: The prompt in JSON format.

#     """
#     data = [prompt_input]
#     data_dict = {
#         "id": "1",
#         "inputs": [
#             {"name": "input0", "shape": [-1], "datatype": "BYTES", "data": data}
#         ],
#     }
#     return data_dict


# Generate a new response if last message is not from assistant
def add_assistant_response():
    """
    Adds the assistant's response to the chat history and displays
    it to the user.

    """
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking..."):
                if LLM_HISTORY == "on":
                    response = generate_chat_response(prompt)
                else:
                    response = generate_response(prompt)
                if not response:
                    st.markdown(
                        "<p style='color:red'>Inference backend is unavailable. "
                        "Please verify if the inference server is running</p>",
                        unsafe_allow_html=True,
                    )
                    return
                if LLM_MODE == "code":
                    st.code(response, language="python")
                else:
                    st.write(response)
        chatmessage = {"role": "assistant", "content": response}
        st.session_state.messages.append(chatmessage)


add_assistant_response()
