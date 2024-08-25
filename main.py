import streamlit as st 
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os 

from dotenv import load_dotenv, dotenv_values 

# loading variables from .env file
load_dotenv() 
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")



from langchain_community.document_loaders import DirectoryLoader
def load_data_from_folders(main_folder_path):
    data = {}
    for category_folder in os.listdir(main_folder_path):
        category_label = category_folder 
        category_path = os.path.join(main_folder_path, category_folder)
        loader = DirectoryLoader(f'{category_path}', glob="**/*.txt" ,  use_multithreading=True , show_progress=True)
        data[category_label]= loader
    return data

data = load_data_from_folders("archive_2")

Culture = data["Culture"].load()



    
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  
)



prompt = """
You are an Ai bot your role is to answer the users questions form the knowledge  that in retriver 
at the end of the answer thank the user 
The answer must be detaild and no less than 100 words.

what to do if the answer is not envluded in the prompt or the context 
    1. apologies to the user.
    2. tell the user that you do not know the answer for the asked question 
    3. ask the user if he has more question to ask. 
    4. do not mention anything about the context.

for the asnwer:
    1. The output must be the answer only without any additional thoughts..

knowledge you know:
{context}

Question : {question}

answer:
"""
class embedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self , docs):
        embeddings = self.model.encode(docs)
        return embeddings.tolist()
    def embed_query(self , query):
        return self.model.encode(query).tolist()




docs = Culture
prompt = ChatPromptTemplate.from_template(prompt)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)




rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
import streamlit as st


st.title("Simple chat")

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


    
    with st.chat_message("assistant"):
        response = st.write(rag_chain.invoke(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

