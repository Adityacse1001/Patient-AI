# Import necessary libraries
from dotenv import load_dotenv  # Load environment variables
load_dotenv()

import streamlit as st  # Streamlit for UI
from langchain_ollama.llms import OllamaLLM  # Ollama LLM for AI-powered responses
from langchain_core.documents import Document  # Document handling
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # Prompt templates for AI
from langchain_community.document_loaders import PDFPlumberLoader  # PDF document loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into chunks
from langchain_community.vectorstores.faiss import FAISS  # FAISS vector store for embeddings
## Embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # BGE embeddings
from langchain_ollama.llms import OllamaLLM  # Ollama LLM model
from langchain_ollama import OllamaEmbeddings  # Ollama embeddings
from langchain.chains import create_retrieval_chain  # Retrieval chain creation
from langchain_core.prompts import MessagesPlaceholder  # Handles chat history placeholders
from langchain_core.messages import HumanMessage, AIMessage  # Message format
from langchain_chains.combine_documents import create_stuff_documents_chain  # Document combination
from langchain_chains.history_aware_retriever import create_history_aware_retriever  # Context-aware retriever

# Global variable to store chat history
global chat_history
chat_history = []

# File storage path
FILE_PATH = 'document_store/pdfs/'

# Function to save the uploaded PDF file
def save_uploaded_file(uploaded_file):
    file_path = FILE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())  # Save file in binary mode
    return file_path

# Function to extract text from a PDF file
def get_document_from_pdf(file_path):
    document_loader = PDFPlumberLoader(file_path)  # Load PDF using PDFPlumber
    docs = document_loader.load()
    
    # Split the document into smaller chunks for better processing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Each chunk will have 300 characters
        chunk_overlap=20  # Overlap to maintain context
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

# Function to create a vector store (database for embeddings)
def create_db(docs):
    embedding = OllamaEmbeddings(model="granite-embedding")  # Use Ollama embeddings
    vectorStore = FAISS.from_documents(docs, embedding)  # Store document embeddings in FAISS
    return vectorStore

# Function to add chat history to the vector store
def embedd_chat_history(query, response, vectorStore):
    new_chat = Document(page_content=f"[Doctor]: {query}\n[Patient]: {response}")
    vectorStore.add_documents([new_chat])  # Add chat as a new document

# Function to create the AI chat response chain
def create_chain(vectorStore):
    # Initialize AI model
    model = OllamaLLM(model="mistral:7b", temperature=0.6)  # LLM with temperature for variability

    # Chat prompt template
    chatprompt = ChatPromptTemplate.from_messages([
        # System instructions for AI to behave as a patient
        ("system",
        "**🔹 You are the patient described in the scenario. Stay in character.**\n\n"
        "- Speak casually, using natural language.\n"
        "- Only mention symptoms you've experienced.\n"
        "- Do not suggest diagnoses or treatments.\n"
        "- Stop responding once the doctor prescribes medicine or tests.\n"
        "- Greet the doctor only in the first message."
        ),

        # Scenario and background knowledge
        ("system",
        "**📋 Scenario (Provided by User):**\n"
        "{context}\n"
        "**🏥 Situation:** You are visiting a doctor because of a health issue."
        ),

        # Chat history to maintain context
        MessagesPlaceholder(variable_name="chat_history"),

        # Initial AI response
        ("human", "{input}"),
        ("ai", "Hello, doctor. I've been feeling off lately and wanted to get it checked.")
    ])

    # Create document response chain
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=chatprompt
    )

    # Create retriever for fetching relevant information from stored embeddings
    retriever = vectorStore.as_retriever(search_kwargs={"K": 2})  # Retrieve top 2 relevant documents

    # Prompt for generating search queries from conversation context
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system",
        "🔍 **Task:** Generate a precise search query based on the conversation.\n"
        "- Ignore greetings and small talk.\n"
        "- Focus on symptoms, medical concerns, and relevant details.\n"
        "- Ensure the query is concise and useful for retrieving relevant information."
        ),
        ("human", "What is the most relevant search query based on this conversation?")
    ])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    # Create final retrieval-based chain
    retriever_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retriever_chain

# Function to process user input and generate a response
def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]

# Function to handle message generation and chat history
def generate_message(chain, user_input, vectorStore):
    response = process_chat(chain, user_input, st.session_state.chat_history)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
    embedd_chat_history(user_input, response, vectorStore)  # Store the conversation in vector DB

    # Store conversation history
    st.session_state.conversation.append({
        "user": user_input,
        "assistant": response
    })

    # Display chat messages
    for entry in st.session_state.conversation:
        messages.chat_message("user", avatar="🧑‍⚕️").write(entry['user'])
        messages.chat_message("answer", avatar="🤖").write(entry['assistant'])

# Main execution starts here
if __name__ == "__main__":
    # Chatbot UI setup
    height = 500
    title = "🤖 Patient Bot"
    icon = "🏥"

    # Initialize session states
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
        
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
        
    if 'chat' not in st.session_state:
        st.session_state.chat = True

    # Function to toggle file upload section
    def toggle_click():
        st.session_state.clicked = not st.session_state.clicked

    # Function to enable/disable chat
    def disable_chat(value):
        st.session_state.chat = value

    # Set page title and icon
    st.set_page_config(page_title=title, page_icon=icon)

    # UI Layout
    col1, col2 = st.columns([4, 1], gap="large")
    with col1:
        st.header(title)
    with col2:
        st.button("Upload Files" if not st.session_state.clicked else "Close Files", on_click=toggle_click)

    # File upload section
    if st.session_state.clicked:
        uploaded_files = st.file_uploader(
            "Upload Your Research Document (PDF)",
            type="pdf",
            help="Select a PDF for analysis",
            accept_multiple_files=False,
        )
        
        if uploaded_files:
            saved_path = save_uploaded_file(uploaded_files)
            docs = get_document_from_pdf(saved_path)
            vectorStore = create_db(docs)
            chain = create_chain(vectorStore)
            disable_chat(True)
            messages = st.container(border=True, height=height)
            disable_chat(False)

    # Chat input field
    if prompt := st.chat_input("Enter your question...", disabled=st.session_state.chat, key="prompt"):
        generate_message(chain, prompt, vectorStore)
