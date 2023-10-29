import re 
import json

from langchain.document_loaders import TextLoader

from base_code.graph_creator import *



PDF_INPUT_PATH = 's3://ryan2008-textract-practice-1/'
TEXT_OUTPUT_PATH = "/Users/ryanmukai/Documents/github/langchain_1/documents/"

text_filename_list = [

        #TEXT_OUTPUT_PATH + "Bylaws2.txt",
        TEXT_OUTPUT_PATH + "CC&Rs2.txt"
        #TEXT_OUTPUT_PATH + "Rules & Regs2.txt",
        #TEXT_OUTPUT_PATH + "Rules for Election2.txt",
        #TEXT_OUTPUT_PATH + "Rules for Voting2.txt",
        #TEXT_OUTPUT_PATH + "CID STATEMENT Hubbard Gardens 03-21-2018 2.txt",
        #TEXT_OUTPUT_PATH + "SI-COMPLETE Hubbard Gardens 03-21-2018 2.txt",
        #TEXT_OUTPUT_PATH + "SI-COMPLETE Hubbard Gardens 03-09-2016 2.txt",
        #TEXT_OUTPUT_PATH + "Articles of Incorporation Hubbard Gardens May 19 1998 2.txt",
        #TEXT_OUTPUT_PATH + "Hubbard Gardens Search - Business Entities California Secretary of State 2.txt",
        #TEXT_OUTPUT_PATH + "Hubbard Gardens HOA policies and budget 2021 2.txt"

]


first_document_flag = True
for current_text_filename in text_filename_list:

    loader = TextLoader(current_text_filename)
    if first_document_flag:
        loader.load_and_split()
        raw_documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
        list_of_pages = text_splitter.split_documents(raw_documents)
        first_document_flag = False
    else:
        loader.load_and_split()
        raw_documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
        list_of_pages = list_of_pages + text_splitter.split_documents(raw_documents)


list_of_docs = [
    {
        "new_document": "Igor is dating Natasha and must buy her flowers from the florist."
    },
    {
        "new_document": "Natasha likes roses and orchids but dislikes carnations."
    },
    {
        "new_document": "The florist has orchids and carnations but not roses."
    },
    {
        "new_document": "Igor has a sister named Svetlana."
    },
    {
        "new_document": "Igor was born in Dnipro, Ukraine."
    },
    {
        "new_document": "Natasha was born in Lviv, Ukraine."
    }
]

#output_list = create_graph_from_documents(list_of_docs)
output_list = create_graph_from_documents(list_of_pages)

with open('test_script.txt','w') as fp:

    for current_item in output_list:
        fp.write("%s\n" % (current_item,))


