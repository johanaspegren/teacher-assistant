from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import TextLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.document_loaders import PyPDFLoader


print("hio")


# This adds documents from a langchain loader to the database. The customized splitters serve to be able to break at sentence level if required.
def add_documents(loader, instance):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators= ["\n\n", "\n", ".", ";", ",", " ", ""])
    texts = text_splitter.split_documents(documents)
    instance.add_documents(texts)

# Create embeddings instance
embeddings = OpenAIEmbeddings(openai_api_key="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")


# Create Chroma instance
instance = Chroma(embedding_function=embeddings, persist_directory="C:\\Users\\johan\\development\\teacher-assistant\\docs")


# add Knowledgebase Dump (CSV file)
# C:\Users\johan\development\teacher-assistant\docs
#loader = TextLoader("C:\\DocBot\\Input\\en-kb@forum-combit-net-2023-04-25.dcqresult.csv")
#add_documents(loader, instance)

# add documentation PDFs
pdf_files = [
    "C:\\Users\\johan\\development\\teacher-assistant\\docs\\dis.pdf"
]

for file_name in pdf_files:
    loader = UnstructuredPDFLoader(file_name)
    loader = PyPDFLoader(file_name)
    add_documents(loader, instance)

 
tech_template = """As a teacher assistant bot, your goal is to answer questions
based on the provided documents. 
{context}

Q: {question}
A: """

PROMPT = PromptTemplate(
    template=tech_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                                                openai_api_key="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"),
                                                chain_type="stuff",
                                                retriever=instance.as_retriever(),
                                                chain_type_kwargs={"prompt": PROMPT})

    def ask(query):
        # process the input string here
        response = qa.run(query)
        return response

    ask("what is the text about?")
