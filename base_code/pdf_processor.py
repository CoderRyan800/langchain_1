from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders import AmazonTextractPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

import io
import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from nanonets import NANONETSOCR

# fp = open('/home/ryanmukai/Documents/github/nanoets_api_key_0.txt','r')
# nanonets_string = fp.readline()
# fp.close()

# nanonets_api_key_0 = nanonets_string.strip()

# def extract_text_from_pdf(pdf_path, output_text_path):

#     model = NANONETSOCR()
#     model.set_token(nanonets_api_key_0)

#     string = model.convert_to_string(pdf_path,formatting='lines and spaces') 
#     # DEBUG
#     print(string)
#     with open(output_text_path, 'w') as f:
#         f.write(string)


import pytesseract
from pdf2image import convert_from_path

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFium2Loader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import PDFPlumberLoader

def extract_pytesseract(pdf_path, output_text_path):

    pages = convert_from_path(pdf_path, 500)
    text = ''

    for page in pages:
        text += pytesseract.image_to_string(page)

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_pytesseract

def extract_PyPDFLoader(pdf_path, output_text_path):

    loader = PyPDFLoader(pdf_path)

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_PyPDFLoader

def extract_AmazonTextractPDFLoader(pdf_path, output_text_path):

    loader = AmazonTextractPDFLoader(pdf_path,region_name='us-east-1')

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_AmazonTextractPDFLoader

def extract_UnstructuredPDFLoader(pdf_path, output_text_path):

    loader = UnstructuredPDFLoader(pdf_path)

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_UnstructuredPDFLoader


def extract_PyPDFium2Loader(pdf_path, output_text_path):

    loader = PyPDFium2Loader(pdf_path)

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_PyPDFium2Loader


def extract_PDFMinerLoader(pdf_path, output_text_path):

    loader = PDFMinerLoader(pdf_path)

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_PDFMinerLoader

def extract_PyMuPDFLoader(pdf_path, output_text_path):

    loader = PyMuPDFLoader(pdf_path)

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_PyMuPDFLoader

def extract_PDFPlumberLoader(pdf_path, output_text_path):

    loader = PDFPlumberLoader(pdf_path)

    pages = loader.load_and_split()
    text = ''

    for page in pages:
        text += page.page_content

    # DEBUG
    print(text)
    with open(output_text_path, 'w') as f:
        f.write(text)

# End function extract_PDFPlumberLoader


class pdf_processor(object):

    def __init__(self, pdf_filename_list, chat_gpt_model_choice='gpt-4',pdf_flag=True):
        """
        Initializer for PDF handling object.  This object can load up a PDF and answer questions about it.

        pdf_filename: Filename with full path of PDF file to process.
        chat_gpt_model_choice: Choice of model, such as 'gpt-3.5-turbo'.
        """

        # Begin by loading the PDF file and splitting it into pages.
        # Do only if pdf_flag is true.
        first_document_flag = True


        # End if logic

        for current_pdf_filename in pdf_filename_list:
            if pdf_flag:
                loader = PyPDFLoader(current_pdf_filename)
            else:
                loader = TextLoader(current_pdf_filename)
            raw_documents = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
            documents = text_splitter.split_documents(raw_documents)
            if first_document_flag:
                self.pages = documents
                first_document_flag = False
            else:
                self.pages = self.pages + documents

        # Next, we must load these pages into a vector store to allow processing.
        # Note we are using OpenAIembeddings since these translate human language into vectors
        # that will make sense to an OpenAI neural network.

        self.vectorstore = Chroma.from_documents(documents=self.pages,embedding=OpenAIEmbeddings())

        # Create a retriever from the vectorstore.

        self.retriever = self.vectorstore.as_retriever()

        self.query_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0)
        self.retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=self.retriever, llm=self.query_llm
        )

        # Create an LLM object we can use.

        self.llm = ChatOpenAI(model_name=chat_gpt_model_choice, temperature=0)

        # Create a memory for conversations.

        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               output_key="answer",
                                               return_messages=True)

        # Creat a chat object.

        self.chat = ConversationalRetrievalChain.from_llm(self.llm, 
                                                          retriever=self.retriever_from_llm, 
                                                          memory=self.memory,
                                                          return_source_documents=True)




    # End __init__ method

    def get_relevant_documents(self, query):
        """
        Responds to a human language query about the documents and fetches relevant documents from
        the vector store.

        query: Human language query about the documents.
        """

        docs = self.vectorstore.similarity_search(query)

        return docs

    # End get_relevant_documents

    def query_the_document(self, query):
        """
        This takes a query and answers questions about the document.  It acts with a memory
        of the conversation since the object was created (stored in self.memory) and also
        with knowledge of the document as the document is vectorized in the self.vectorstor.

        query: Human language statements or questions in the conversation about the document.
        """

        result = self.chat({"question": query})

        return result

    # End method query_the_document

# End class declaration pdf_processor

def convert_to_json_serializable(input_response):

    if type(input_response) is dict:
        new_dict = dict()
        for current_key in input_response:
            new_dict[current_key] = convert_to_json_serializable(input_response[current_key])
        return new_dict
    elif type(input_response) is list:
        new_list = []
        for current_item in input_response:
            new_list.append(convert_to_json_serializable(current_item))
        return new_list
    else:
        try:
            new_item = input_response.json()
        except:
            new_item = input_response
        return new_item

# End convert_to_json
