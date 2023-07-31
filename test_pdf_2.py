import json

from code.pdf_processor import *

proc_obj = pdf_processor(pdf_filename="/home/ryanmukai/Documents/github/redesigned-octo-goggles/writing/lstm_paper.pdf",
                         chat_gpt_model_choice='gpt-4')

query = "Please advise on how this document can be written for a peer reviewed journal: PlosONE.  Include a detailed outline for a re-written version of this document with a detailed list of steps to follow for the re-write."

response = proc_obj.query_the_document(query)

fp = open('paper_suggestions_v1.txt','w')

fp.write(response['answer'])

fp.close()

query = "Please create a re-written abstract to help me get started."

response = proc_obj.query_the_document(query)

fp = open('paper_suggestions_v2.txt','w')

fp.write(response['answer'])

fp.close()
