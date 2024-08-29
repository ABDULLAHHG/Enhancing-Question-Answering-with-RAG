import streamlit as st 
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
import os 

from dotenv import load_dotenv, dotenv_values 
st.sidebar.write("")





# if u Havent installed them 
# !pip install unstructured
# !pip install python-magic-bin
# !pip install -U sentence-transformers
# !pip install -qU "langchain-chroma>=0.1.2"
# !pip install streamlit 
# !pip install langchain-google-vertexai




class embedding:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # self.model = SentenceTransformer("intfloat/multilingual-e5-large")
    def embed_documents(self , docs):
        embeddings = self.model.encode(docs)
        return embeddings.tolist()
    def embed_query(self , query):
        return self.model.encode(query).tolist()




@st.cache_data
def load_data_from_folders(main_folder_path , catigory_type):
    data = {}
    for category_folder in os.listdir(main_folder_path):
        category_label = category_folder 
        category_path = os.path.join(main_folder_path, category_folder)
        loader = DirectoryLoader(f'{category_path}', glob="**/*.txt" ,  use_multithreading=True , show_progress=True)
        data[category_label]= loader
        
    if catigory_type in data.keys():
        return data[catigory_type].load()
    else:
        whole_data = []
        for key in data.keys:
            whole_data.extend(data[key].load())
        return whole_data

def response_llm(prmpt , catigory_type):
    Culture = load_data_from_folders("archive_2", catigory_type)



    docs = Culture


    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)



    prompt = """
You are an Ai bot your role is to answer the users questions form the knowledge
at the end of the answer thank the user
The answer must be detaild

what to do if the answer is not envluded in the prompt or the context
    1. apologies to the user.
    2. tell the user that you do not know the answer for the asked question
    3. ask the user if he has more question to ask.
    4. do not mention anything about the context.

for the asnwer:
    1. The output must be the answer only without any additional thoughts..
    2. do not answer with anything that is not in knowledge
    3. do not add extra words to response that is not in knowledge
    4. answer with arabic language only

knowledge you know:
{context}

Question : {question}

answer:
"""

    prompt = ChatPromptTemplate.from_template(prompt)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(prmpt)



# loading variables from .env file
load_dotenv() 
try: 
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

            
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  
)
    st.title("Simple chat")
    options =(os.listdir("archive_2"))
    options.append("Full Data")
    type = st.sidebar.selectbox(label="Select data" ,options = options, index =0)
    if type=="Full Data":
        st.sidebar.text("its will take a very long time.\nSend a message than the data will load it self")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        
        with st.chat_message("assistant"):
            response = st.write(response_llm(prompt , type))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

except TypeError:
    st.error("Dont forget to create a file called .env then add a variable called `GOOGLE_API_KEY`=`Your api key` good luck")

