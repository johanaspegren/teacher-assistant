import os

from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = "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"


root_dir = "C:\\Users\\johan\\development\\teacher-assistant\\docs\\"

pdf_folder_path = f'{root_dir}'
print(os.listdir(pdf_folder_path))

from langchain.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embedding=embeddings, 
                                 persist_directory=root_dir)


vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8) , vectordb.as_retriever(), memory=memory)

def ask(query):
    result = pdf_qa({"question": query})
    print("Answer:" + result["answer"])
    

ask("What is disruptive innovation?")
ask("Who coined the term disruptive innovation, and when?")

ask("Can you ask me a question about the important parts of the text?")

ask("Can you ask me a question and check my answer?")
