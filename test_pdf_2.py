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

fp = open('paper_suggestions_abstract.txt','w')

fp.write(response['answer'])

fp.close()

query = "Please state the last two questions that I asked you?"

response = proc_obj.query_the_document(query)

fp = open('response_to_simple_question.txt','w')

fp.write(response['answer'])

fp.close()

query = "Please provide a starter introduction to aid me in getting started."

response = proc_obj.query_the_document(query)

fp = open('paper_suggestions_intro.txt','w')

fp.write(response['answer'])

fp.close()

query = "Is this research actually innovative, or is it only a rehash of what already exists?  I know you don't have knowledge beyond a certain date but please try to answer.  If this is innovative, what is the key innovation?  If it is not, what other literature already covers knowledge of own state in a neural network that makes this redundant?"

response = proc_obj.query_the_document(query)

fp = open('assessment_of_whether_this_is_innovative.txt','w')

fp.write(response['answer'])

fp.close()

json_string = json.dumps(json.loads(proc_obj.memory.json()),indent=4)

fp = open('memory.json','w')

fp.write(json_string)

fp.close()
