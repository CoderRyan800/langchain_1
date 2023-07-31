from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class pdf_processor(object):

    def __init__(self, pdf_filename, chat_gpt_model_choice='gpt-4'):
        """
        Initializer for PDF handling object.  This object can load up a PDF and answer questions about it.

        pdf_filename: Filename with full path of PDF file to process.
        chat_gpt_model_choice: Choice of model, such as 'gpt-3.5-turbo'.
        """

        # Begin by loading the PDF file and splitting it into pages.

        loader = PyPDFLoader(pdf_filename)
        self.pages = loader.load_and_split()

        # Next, we must load these pages into a vector store to allow processing.
        # Note we are using OpenAIembeddings since these translate human language into vectors
        # that will make sense to an OpenAI neural network.

        self.vectorstore = Chroma.from_documents(documents=self.pages,embedding=OpenAIEmbeddings())

        # Create a retriever from the vectorstore.

        self.retriever = self.vectorstore.as_retriever()

        # Create an LLM object we can use.

        self.llm = ChatOpenAI(model_name=chat_gpt_model_choice, temperature=0)

        # Create a memory for conversations.

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Creat a chat object.

        self.chat = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.retriever, memory=self.memory)

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

