import json

from base_code.pdf_processor import *

proc_obj = pdf_processor(pdf_filename_list=[
    "/home/ryanmukai/Documents/github/langchain_1/other/Descanso14_MSL_Telecom.pdf"
    ],
    chat_gpt_model_choice='gpt-4')

query = "During which mission phases will UHF be used?"
response = proc_obj.query_the_document(query)
query = "Which radio is used for UHF communications?"
response = proc_obj.query_the_document(query)
query = "Which radio is used for surface X-Band communications?"
response = proc_obj.query_the_document(query)
query = "Please describe the Electra Lite radio."
response = proc_obj.query_the_document(query)
query = "Please describe the SDST."
response = proc_obj.query_the_document(query)

json_string = json.dumps(json.loads(proc_obj.memory.json()),indent=4)

fp = open('/home/ryanmukai/Documents/github/langchain_1/other/memory.json','w')

fp.write(json_string)

fp.close()
