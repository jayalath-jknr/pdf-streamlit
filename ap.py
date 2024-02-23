from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


def read_pdf():
    loader = PyPDFLoader("pdf/HANDBOOK_ENGLISH.pdf")
    pages = loader.load_and_split()
    print(pages[0])
    return pages[:2]

def create_embeddings(text) :
    # Replace with your OpenAI API key
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
    OPENAI_API_KEY = "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"
    faiss_index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    faiss_index2 = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    print(faiss_index)
    print(faiss_index2)
    # docs = faiss_index.similarity_search("What are the different subject streams that are available for  B. Sc. MIT (Management and Information technology) students University of Kelaniya to select after 3rd year", k=2)
    # for doc in docs:
    #     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

def main():
    text = read_pdf()      
    create_embeddings(text)

if __name__ == "__main__":
    main()
        