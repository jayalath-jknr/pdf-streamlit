import validators
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader
import os
import json
import requests
import sys, pathlib, fitz
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# openai.api_key = os.getenv("OPENAI_API_KEY")

def read_pdf():
    print("Reading the PDF")
    reader = PdfReader("pdf/HANDBOOK_ENGLISH.pdf")
    print("Number of pages:", len(reader.pages))
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()
    print("Text from page 1:", text)
    return text

def create_text_file () :
    fname = "pdf/HANDBOOK_ENGLISH.pdf"  # get document filename
    with fitz.open(fname) as doc:  # open document
        text = chr(12).join([page.get_text() for page in doc])
    # write as a binary file to support non-ASCII characters
    pathlib.Path(fname + ".txt").write_bytes(text.encode())

def create_embeddings(text) :
    # # I will use to load the PDF from the path "pdf/HANDBOOK_ENGLISH.pdf.txt"
    
    # loader = UnstructuredURLLoader("pdf/HANDBOOK_ENGLISH.pdf.txt")
    # with open("pdf/HANDBOOK_ENGLISH.pdf.txt") as f:
    #     hand_book= f.read()
    # loader = TextLoader("pdf/HANDBOOK_ENGLISH.pdf.txt")
    # documents = loader.load()
    # # I will use the CharacterTextSplitter to split the text into paragraphs
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )   
    # I will use the OpenAIEmbeddings to embed the paragraphs
    # texts = splitter.create_documents([hand_book])
    # print(texts[0])
    docs = splitter.split_documents(text)

    embeddings = OpenAIEmbeddings(openai_api_key="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")
    # I will use the FAISS to store the embeddings
    vector_store = FAISS.from_texts( texts= docs, embedding=embeddings)
    # # I will use the ChatOpenAI to answer the questions
    # chat_model = ChatOpenAI()
    # # I will use the ConversationBufferMemory to store the conversation
    # memory = ConversationBufferMemory()
    # # I will use the ConversationalRetrievalChain to chain the components
    # chain = ConversationalRetrievalChain(loader, splitter, embeddings, vector_store, chat_model, memory)
    # return chain

#   Add main function to call the functions
def main():
    docs = read_pdf()
    create_embeddings(docs)


if __name__ == "__main__":
    main()