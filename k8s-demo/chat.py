"""
GPT-in-a-Box Streamlit App
This module defines a Streamlit app for interacting with different Large Language models.
"""

import os
import json
import sys
import requests
import streamlit as st
import subprocess
import time

# Add supported models to the list
AVAILABLE_MODELS = ["tiny-llama"]
#AVAILABLE_MODELS = ["llama2-7b-chat", "codellama-7b-python"]
# AVAILABLE_MODELS = ["llama2-7b", "mpt-7b" , "falcon-7b"]
BASE_DIR = os.environ["CHAT_DIR"] + "/k8s-demo/"
ASSISTANT_SVG = BASE_DIR + "assistant.svg"
USER_SVG = BASE_DIR + "user.svg"
LOGO_SVG = BASE_DIR + "nutanix.svg"

LLM_MODE = "chat"
LLM_HISTORY = "off"

# Read deployment name from file to get values to construct URL
# try:
#     with open("config.txt", "r") as f:
#         for line in f:
#             key, value = line.strip().split('=')
#             if key == "DEPLOY_NAME":
#                 DEPLOY_NAME = value
#         #print(f"Using deployment {DEPLOY_NAME}")
# except Exception as e:
#     st.error(f"{e}")
#     st.stop()

DEPLOY_NAME="tiny-llama-deploy"

get_svc_hostname_cmd=f'kubectl get inferenceservice {DEPLOY_NAME} '
get_svc_hostname_cmd+='-o jsonpath=\'{.status.url}\' | cut -d "/" -f 3'
#print(get_svc_hostname_cmd)
svc = subprocess.run(get_svc_hostname_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

get_ingress_host_cmd="kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}'"
#print(get_ingress_host_cmd)
host = subprocess.run(get_ingress_host_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

get_port_cmd="kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name==\"http2\")].nodePort}'"
#print(get_port_cmd)
port = subprocess.run(get_port_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if svc.stderr or host.stderr or port.stderr:
    st.error(f"Encountered 1 or more errors when running kubectl commands, please check that your KUBECONFIG is valid and your cluster is running \n" \
    f"{svc.stderr}\n" \
    f"{host.stderr}\n" \
    f"{port.stderr}\n")
    st.stop()

SERVICE_HOSTNAME = svc.stdout.strip()
INGRESS_HOST = host.stdout.strip()
INGRESS_PORT = port.stdout.strip()

if not os.path.exists(ASSISTANT_SVG):
    ASSISTANT_AVATAR = None
else:
    ASSISTANT_AVATAR = ASSISTANT_SVG

if not os.path.exists(USER_SVG):
    USER_AVATAR = None
else:
    USER_AVATAR = USER_SVG

html_code = """
<div style="padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
Note: This lab is for demo purposes only to show connecting a chatbot app to a running LLM. The lab is not using GPU, so responses may take up to a minute and will likely produce lower quality responses than running a larger model on GPU.</div>
"""

# App title
st.title("Welcome to GTS '24")
st.subheader("Powered by Nutanix Kubernetes Engine")
st.markdown(html_code, unsafe_allow_html=True)

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
        "GPT-in-a-Box is a turnkey AI solution for organizations wanting to implement GPT "
        "capabilities while maintaining control of their data and applications. Read the "
        "[announcement]"
        "(https://www.nutanix.com/blog/nutanix-simplifies-your-ai-innovation-learning-curve)"
    )

    st.subheader("Models")
    selected_model = st.sidebar.selectbox(
        "Choose a model", AVAILABLE_MODELS, key="selected_model"
    )
    if selected_model == "tiny-llama":
        LLM = "tiny-llama"
        LLM_HISTORY = "on"
        st.markdown("[Tiny Llama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) "
                "is a compact model with 1.1B model that can run on CPU. "
                "It uses the same architecture and tokenizer as Llama 2. It was finetuned "
                "on a variant of the UltraChat dataset."
        )
    elif selected_model == "llama2-7b":
        LLM = "llama2_7b"
        st.markdown(
            "Llama2 is a state-of-the-art foundational large language model which was "
            "pretrained on publicly available online data sources. This chat model "
            "leverages publicly available instruction datasets and over 1 "
            "million human annotations."
        )
    elif selected_model == "mpt-7b":
        LLM = "mpt_7b"
        st.markdown(
            "MPT-7B is a decoder-style transformer with 6.7B parameters. It was trained "
            "on 1T tokens of text and code that was curated by MosaicML’s data team. "
            "This base model includes FlashAttention for fast training and inference and "
            "ALiBi for finetuning and extrapolation to long context lengths."
        )
    elif selected_model == "falcon-7b":
        LLM = "falcon_7b"
        st.markdown(
            "Falcon-7B is a 7B parameters causal decoder-only model built by TII and "
            "trained on 1,500B tokens of RefinedWeb enhanced with curated corpora."
        )
    elif selected_model == "codellama-7b-python":
        LLM = "codellama_7b_python"
        LLM_MODE = "code"
        st.markdown(
            "Code Llama is a large language model that can use text prompts to generate "
            "and discuss code. It has the potential to make workflows faster and more "
            "efficient for developers and lower the barrier to entry for people who are "
            "learning to code."
        )
    elif selected_model == "llama2-7b-chat":
        LLM = "llama2_7b_chat"
        LLM_HISTORY = "on"
        st.markdown(
            "Llama2 is a state-of-the-art foundational large language model which was "
            "pretrained on publicly available online data sources. This chat model "
            "leverages publicly available instruction datasets and over 1 million "
            "human annotations."
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
    url = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v2/models/{LLM}/infer"
    headers = {"Host": SERVICE_HOSTNAME, "Content-Type": "application/json; charset=utf-8"}
    try:
        start = time.perf_counter()
        response = requests.post(url, json=input_prompt, timeout=600, headers=headers)
        request_time = time.perf_counter() - start
        print(request_time)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        print("Error in requests: ", url)
        return ""
    output_dict = json.loads(response.text)
    output = output_dict["outputs"][0]["data"][0]
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

    if LLM == "tiny-llama":
        string_dialogue = (
            "<|system|>"
            "You are a helpful assistant.</s>"
        )
    else:
        string_dialogue = (
            "[INST] <<SYS>> You are a helpful assistant. "
            " You answer the question asked by 'User' once"
            " as 'Assistant'. <</SYS>>[/INST]" + "\n\n"
        )

    for dict_message in st.session_state.messages[:-1]:
        if dict_message["role"] == "user":
            if LLM == "tiny-llama":
                string_dialogue += "<|user|>\n " + dict_message["content"] + "</s>" 
            else:
                string_dialogue += "User: " + dict_message["content"] + "[/INST]" + "\n\n"
        else:
            if LLM == "tiny-llama":
                string_dialogue += "<|assistant|>\n"
            else:
                string_dialogue += (
                    "Assistant: " + dict_message["content"] + " [INST]" + "\n\n"
                )
    if LLM == "tiny-llama":
        string_dialogue += "<|user|>\n " + f"{input_prompt}" + "\n\n"
        input_text = f"{string_dialogue}" + "\n\n" + "<|assistant|>\n"
    else:
        string_dialogue += "User: " + f"{input_prompt}" + "\n\n"
        input_text = f"{string_dialogue}" + "\n\n" + "Assistant: [/INST]"

    output = generate_response(input_text)
    # Generation failed
    if len(output) <= len(input_text):
        return ""
    print(f"\nInput: {input_text}")
    #print(f"\nOutput: {output}")

    # Tiny Llama Specific
    if LLM == "tiny-llama":
        # We want the portion of text after the last instance of <|assistant|> in the output
        substring = "<|assistant|>"
        last_index = output.rfind(substring)
        if last_index != -1:
            response = output[last_index + len(substring):]
        else:
            st.error("Output not in expected format.")
    else:
        response = output[len(input_text) :]
    
    return response


# User-provided prompt
if prompt := st.chat_input("Ask your query"):
    message = {"role": "user", "content": prompt}
    st.session_state.messages.append(message)
    add_message(message)


def get_json_format_prompt(prompt_input):
    """
    Converts the input prompt into the JSON format expected by the LLM.

    Parameters:
    - prompt_input (str): The input prompt.

    Returns:
    - dict: The prompt in JSON format.

    """
    data = [prompt_input]
    data_dict = {
        "id": "1",
        "inputs": [
            {"name": "input0", "shape": [-1], "datatype": "BYTES", "data": data}
        ],
    }
    return data_dict


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

