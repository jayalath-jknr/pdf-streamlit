import validators
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
import os
import json
import requests
import unstructured
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# from langchain.chains.summarize import load_summarize_chain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# Streamlit app
st.subheader('Ask Questions from PDF')
# This app can Be used to ask questions from a PDF by uploading it to the app
# The app will then extract the text from the PDF and use it to answer the questions

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
# I want to use the PDF to answer the questions
# therefore I want to preprocess the document and extract the text
# I will use the UnstructuredURLLoader to load the PDF
# and the CharacterTextSplitter to split the text into paragraphs
# and the OpenAIEmbeddings to embed the paragraphs
# and the FAISS to store the embeddings
# and the ChatOpenAI to answer the questions

# loading the PDF
if pdf_file:
    # I will use the UnstructuredURLLoader to load the PDF
    loader = UnstructuredURLLoader()
    # I will use the CharacterTextSplitter to split the text into paragraphs
    splitter = CharacterTextSplitter()
    # I will use the OpenAIEmbeddings to embed the paragraphs
    embeddings = OpenAIEmbeddings()
    # I will use the FAISS to store the embeddings
    vector_store = FAISS()
    # I will use the ChatOpenAI to answer the questions
    chat_model = ChatOpenAI()
    # I will use the ConversationBufferMemory to store the conversation
    memory = ConversationBufferMemory()
    # I will use the ConversationalRetrievalChain to chain the components
    chain = ConversationalRetrievalChain(loader, splitter, embeddings, vector_store, chat_model, memory)
    

    

# s section, we set the user authentication, user and app ID, model details, and the URL of 
# the text we want as an input. Change these strings to run your own example.
######################################################################################################

# Your PAT (Personal Access Token) can be found in the portal under Authentification
PAT = 'b509d5dd7edc468b91784dbbc481c38e'
# Specify the correct user_id/app_id pairings
# Since you're making inferences outside your app's scope
USER_ID = 'openai'
APP_ID = 'chat-completion'
# Change these to whatever model and text URL you want to use
MODEL_ID = 'gpt-4-vision-alternative'
MODEL_VERSION_ID = '12b67ac2b5894fb9af9c06ebf8dc02fb'
RAW_TEXT = 'I love your product very much'
# To use a hosted text file, assign the url variable
# TEXT_FILE_URL = 'https://samples.clarifai.com/negative_sentence_12.txt'
# Or, to use a local text file, assign the url variable
# TEXT_FILE_LOCATION = 'YOUR_TEXT_FILE_LOCATION_HERE'

############################################################################
# YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
############################################################################

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (('authorization', 'Key ' + PAT),)

userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

# To use a local text file, uncomment the following lines
# with open(TEXT_FILE_LOCATION, "rb") as f:
#    file_bytes = f.read()

post_model_outputs_response = stub.PostModelOutputs(
    service_pb2.PostModelOutputsRequest(
        user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
        model_id=MODEL_ID,
        version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    text=resources_pb2.Text(
                        raw=RAW_TEXT
                        # url=TEXT_FILE_URL
                        # raw=file_bytes
                    )
                )
            )
        ]
    ),
    metadata=metadata
)
if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
    print(post_model_outputs_response.status)
    raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

# Since we have one input, one output will exist here
output = post_model_outputs_response.outputs[0]

print("Completion:\n")
print(output.data.text.raw)
st.text_area("Completion:", value=output.data.text.raw, height=200)