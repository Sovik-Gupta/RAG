import streamlit as st
import os
import subprocess

# Set Streamlit app background to black
st.markdown(
    """
    <style>
        .reportview-container {
            background: black;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def installPackages(selected_package: str) -> None:
    """
    Install packages based on the selected package.

    Parameters:
    - selected_package (str): The selected package ("LlamaIndex" or "LangChain").

    Returns:
    - None
    """
    if selected_package == "LlamaIndex":
        os.system("pip install -r LlamaIndex_req.txt")
    elif selected_package == "LangChain":
        os.system("pip install -r LangChain_req.txt")

def importingLlamaIndexPackages() -> None:
    """
    Import necessary packages for LlamaIndex.

    Returns:
    - None
    """
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core.prompts.prompts import SimpleInputPrompt
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from llama_index.legacy.embeddings.langchain import LangchainEmbedding

def buildLlamaIndexModel(system_prompt, HuggingFaceLLM):
    """
    Build a LlamaIndex model.

    Parameters:
    - system_prompt: The system prompt.
    - HuggingFaceLLM: The Hugging Face LLM model.

    Returns:
    - llm: The built LlamaIndex model.
    """
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto"
    )
    return llm

def serviceContext(llm, embed_model, ServiceContext):
    """
    Create a service context.

    Parameters:
    - llm: The LLM model.
    - embed_model: The embedding model.
    - ServiceContext: The service context.

    Returns:
    - service_context: The created service context.
    """
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    return service_context

def query(query_engine):
    """
    Query the engine.

    Parameters:
    - query_engine: The query engine.

    Returns:
    - response: The query response.
    """
    command = st.text_input("Enter prompt:")
    response = query_engine.query(command)
    
    return response

def LlamaModel(uploaded_file) -> None:
    """
    Run the Llama model.

    Parameters:
    - uploaded_file: The uploaded file.

    Returns:
    - None
    """
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core.prompts.prompts import SimpleInputPrompt
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from llama_index.legacy.embeddings.langchain import LangchainEmbedding

    documents = uploaded_file
    system_prompt = """
        you are a QA assistant. your goal is to answer question as accuratlt as possible based on 
        the instructions and context provided
        """
    # Default prompt supported by llama2
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")
    llm = buildLlamaIndexModel(system_prompt, HuggingFaceLLM)
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    service_context = serviceContext(llm, embed_model, ServiceContext)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()
    response=query_engine.query("what is Hypothesis Testing")
    # TODO : We can take response from the user on the StreamLit Application itself to see the dynamic output
    # response = query(query_engine)
    st.write(response)
    return response

# Streamlit app title
st.title("Retrieval Augmented Generation (RAG)")

# File uploader for PDF document
uploaded_file = st.file_uploader("Upload PDF Document:", type=["pdf"])

# Generate response
if uploaded_file is not None:
    input_text = uploaded_file.read()  # Read uploaded PDF document
    st.write("Uploaded Document:")
    st.write(input_text[:60])

    st.write(''' Steps we are going to do ahead: 
    1. Read the external text file and split it into chunks.
    2. Initialize an embedding model.
    3. Generate embeddings for each chunk.
    4. Generate embedding of the query ( QE).
    5. Generate a similarity score betweent QE and each of the Chunk embeddings.
    6. Extract top K chunks based on similarity score.
    7. Frame a prompt with the query and the top-k chunks.
    8. Prompt an LLM with the prompt framed in step-7.
    ''')

# Dropdown menu to select package
#  TODO : LangChain is not incorporated right now, only LlamaIndex is in working state.
selected_package = st.selectbox("Select Package:", ["Select Model","LlamaIndex", "LangChain"])
# Download selected package
if selected_package == "LlamaIndex" or selected_package == "LangChain":
    if st.button("Download Package"):
        installPackages(selected_package)

# TODO : Time variable which will wait for the process to complete and then start the other process
if selected_package == "LlamaIndex":
    # Input for the command to run
    st.title("Enter Your HuggingFace API Key")
    command = st.text_input("Enter command:")

    # Run the command
    if st.button("Run Command"):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        st.write("Go back to terminal and paste you HuggingFaceAPI Key")
        st.write("Command Output:")
        st.text_area("Output", result.stdout, height=200)
        if result.stderr:
            st.write("Error Output:")
            st.text_area("Error", result.stderr, height=100)

model = st.button("Run Model")
if selected_package == "LlamaIndex" and model:
    st.write("Lets begin the Magic")
    LlamaModel(uploaded_file)


##################################################################    HAPPY HACKING  ################################################################################
