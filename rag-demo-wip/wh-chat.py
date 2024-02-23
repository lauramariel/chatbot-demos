# Wolfgang

import streamlit as st
from streamlit_chat import message
from KserveML import KserveML
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from torch import cuda


st.set_page_config(page_title='Llama2-Chatbot')
st.header('LLama2 / Milvus enabled Chatbot running on NKE :robot_face:')

@st.cache_resource()
def load_llm():
    endpoint_url = (
            "http://10.55.88.132/v2/models/tiny-llama/infer"
    )
    llm = KserveML(
      endpoint_url=endpoint_url
    )

    return llm


@st.cache_resource()
def load_vector_store():
    modelPath="sentence-transformers/all-mpnet-base-v2"

    device = f'cuda' if cuda.is_available() else 'cpu'

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        cache_folder='/tmp',
        encode_kwargs=encode_kwargs
    )


    # openthe vector store database
    vector_db = Milvus(
        embeddings,
        collection_name = 'nutanixbible_web',
        connection_args={"host": "10.55.88.135", "port": "19530"},
    )
    return vector_db

def create_qa_chain():

    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store()

    prompt_template = """Answer question from context below.
    context: {context}
    question: {question}
    Answer:"""
    llm_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=False,
                                        chain_type_kwargs={'prompt': llm_prompt})

    return qa_chain

def generate_response(query, qa_chain):

    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']


def get_user_input():

    # get the user query
    input_text = st.text_input('I have been trained on Nutanix Bible... ask me anything', "", key='input')
    return input_text



# create the qa_chain
qa_chain = create_qa_chain()


# create empty lists for user queries and responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# get the user query
user_input = get_user_input()


if user_input:

    # generate response to the user input
    response = generate_response(query=user_input, qa_chain=qa_chain)

    # add the input and response to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)


# display conversaion history (if there is one)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')