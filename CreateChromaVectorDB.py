import os 
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import shutil



# devide each catigory to 2 batch function
# Note : devide each catigory to 2 batch wont change the size of the file 
# def load_data_from_folders(main_folder_path , catigory_type):
#     for category_folder in os.listdir(main_folder_path):
#         category_label = category_folder 
#         category_path = os.path.join(main_folder_path, category_folder)

#         data = []
#         batch = 0 
#         for i , txt in  enumerate(os.listdir(category_path) , 1):
#           loader = TextLoader(f"{category_path}/{txt}")
#           data.extend(loader.load())
#           if i%(len(os.listdir(category_path))/2)==0:
#              print(i)
#              save_to_chroma(data , f"chroma/{category_label}-{batch}")
#              batch+=1 
#              data = []
#              print(f"chroma/{category_label}-{batch}")
             
    

def load_data_from_folders(main_folder_path , catigory_type):
    data = {}
    for category_folder in os.listdir(main_folder_path):
        category_label = category_folder 
        category_path = os.path.join(main_folder_path, category_folder)
        loader = DirectoryLoader(f'{category_path}', glob="**/*.txt" ,  use_multithreading=True , show_progress=True)
        data[category_label]= loader
        
    for key in data.keys():
      save_to_chroma(data[key].load(), f"chroma/{key}/")



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



# Path to the directory to save Chroma database

def save_to_chroma(chunks: list[Document] , CHROMA_PATH:str):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    embedding(),
    persist_directory=CHROMA_PATH
  )

  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
  """
  Function to generate vector database in chroma from documents.
  """
  documents = load_data_from_folders("archive_2" , "whole") # Load documents from a source


generate_data_store()
