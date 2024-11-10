import os

from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# load the environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
print(working_dir)




# loads document from file path using unstructured pdf loader and extracts text from it.
#This method returns a list of Document objects, each representing a page or section of the PDF.
def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents




#converts documents into chunks, chunks are converted to vectors and faiss indexes these vectors.
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore



# it builds a chain of llm, vector store and memory of past chat.
def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain




st.set_page_config(
    page_title="Chat with Patient Medical Records",
    page_icon="📄",
    layout="centered"
)



st.title(" ⚕️ Chat with Patient Medical Records - LLAMA 3.1")



# initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



uploaded_file = st.file_uploader(label="Upload your pdf file", type=["pdf"])


#After user uploads a file this is true and if condition will run
if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    #save the file in backend
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    #creates vector database if it doesn't exists
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    # creates conversation chain if it doesn't exists
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)



for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# This is chat box
user_input = st.chat_input("Ask Llama...")



if user_input:
    # This section saves and displays the user’s message.
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)


    #This section generates a response from the assistant and displays it in the chat.
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
