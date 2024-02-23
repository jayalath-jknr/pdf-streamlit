from PyPDF2 import PdfReader
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Replace with your OpenAI API key
OPENAI_API_KEY = "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"
# Define your chosen sectioning method (sentence or paragraph)
# splitter = SentenceTextSplitter()  # Use SentenceTextSplitter for sentences
splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    ) 
def extract_text(pdf_path):
  loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
  pages = loader.load()
  pages[4].page_content
  return pages[4].page_content

def split_into_sections(text, splitter):
  sections = splitter.split_documents(text) 
  return sections

def create_embeddings(text):
  embeddings_api = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
  embeddings = embeddings_api.create_embeddings(text=text)
  return embeddings

def build_vector_database(sections, embeddings):
  vectorstore = FAISS.from_documents(sections, embeddings)
  return vectorstore

def main():
  pdf_path = "pdf/HANDBOOK_ENGLISH.pdf"
  text = extract_text(pdf_path)
  sections = split_into_sections(text, splitter)
  embeddings = [create_embeddings(section) for section in sections]

  vector_database = build_vector_database(sections, embeddings)

  # Use the vector database for your desired application (e.g., search, similarity)

if __name__ == "__main__":
  main()
