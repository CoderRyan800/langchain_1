import json
import re
from base_code.pdf_processor import *


PDF_INPUT_PATH = 's3://ryan2008-textract-practice-1/'
TEXT_OUTPUT_PATH = "/Users/ryanmukai/Documents/github/langchain_1/documents/"

#MAIN_PATH = "C:/Users/ryan2/Documents/github/langchain_1/documents/"


input_pdf_list = [
        PDF_INPUT_PATH + "Bylaws.pdf",
        PDF_INPUT_PATH + "CC&Rs.pdf",
        PDF_INPUT_PATH + "Rules & Regs.pdf",
        PDF_INPUT_PATH + "Rules for Election.pdf",
        PDF_INPUT_PATH + "Rules for Voting.pdf",
        PDF_INPUT_PATH + "CID STATEMENT Hubbard Gardens 03-21-2018.pdf",
        PDF_INPUT_PATH + "SI-COMPLETE Hubbard Gardens 03-21-2018.pdf",
        PDF_INPUT_PATH + "SI-COMPLETE Hubbard Gardens 03-09-2016.pdf",
        PDF_INPUT_PATH + "Articles of Incorporation Hubbard Gardens May 19 1998.pdf",
        PDF_INPUT_PATH + "Hubbard Gardens Search - Business Entities California Secretary of State.pdf",
        PDF_INPUT_PATH + "Hubbard Gardens HOA policies and budget 2021.pdf"
    ]

output_pdf_list = [

        TEXT_OUTPUT_PATH + "Bylaws2.txt",
        TEXT_OUTPUT_PATH + "CC&Rs2.txt",
        TEXT_OUTPUT_PATH + "Rules & Regs2.txt",
        TEXT_OUTPUT_PATH + "Rules for Election2.txt",
        TEXT_OUTPUT_PATH + "Rules for Voting2.txt",
        TEXT_OUTPUT_PATH + "CID STATEMENT Hubbard Gardens 03-21-2018 2.txt",
        TEXT_OUTPUT_PATH + "SI-COMPLETE Hubbard Gardens 03-21-2018 2.txt",
        TEXT_OUTPUT_PATH + "SI-COMPLETE Hubbard Gardens 03-09-2016 2.txt",
        TEXT_OUTPUT_PATH + "Articles of Incorporation Hubbard Gardens May 19 1998 2.txt",
        TEXT_OUTPUT_PATH + "Hubbard Gardens Search - Business Entities California Secretary of State 2.txt",
        TEXT_OUTPUT_PATH + "Hubbard Gardens HOA policies and budget 2021 2.txt"

]

N = len(input_pdf_list)

#for index in range(N):
#
#    current_input_filename = input_pdf_list[index]
#    current_output_filename = output_pdf_list[index]
#    # extract_UnstructuredPDFLoader(current_input_filename, current_output_filename)
#    extract_AmazonTextractPDFLoader(current_input_filename, current_output_filename)

# gpt-3.5-turbo-16k

proc_obj = pdf_processor(pdf_filename_list=output_pdf_list,
    chat_gpt_model_choice='gpt-3.5-turbo-16k',pdf_flag=False)

question_list = [
    "What is the name of the homeowner's association?",
    "When was it incorporated?",
    "How many cats could a homeowner have?",
    "What items are insured by the association?",
    "What items are homeowners responsible for insuring?",
    "Does the association have earthquake insurance?",
    "What happens if a guest damages the common areas?",
    "How are fire sprinkler systems maintained?",
    "What are the monthly dues for this HOA?",
    "What happens if a homeowner fails to pay the HOA dues?",
    "What are the possible fines the HOA can assess?",
    "How does the HOA handle a homeowner who repeatedly refuses to obey rules or pay fines and assessments?",
    "Are the CC&Rs and bylaws for the HOA confidential?",
    "How can the HOA rule regarding number of cats allowed be changed?",
    "If the HOA does not supply earthquake insurance, then how can I safeguard my unit in case of an earthquake?"
]

response_list = []

fp = open('/Users/ryanmukai/Documents/github/langchain_1/other/response_data.json','w')

stop_flag = False
regex_stop = re.compile('STOP')

while not stop_flag:
    query = input("Please enter your query regarding the HOA documents: ")
    if regex_stop.search(query) is not None:
        break
    response = proc_obj.query_the_document(query)
    response_dict = convert_to_json_serializable(response)
    output_string = json.dumps(response_dict,indent=4)
    fp.write(output_string)
    fp.write("\n")
    print(response_dict['answer'])
    print("\n\n")

fp.close()

json_string = json.dumps(json.loads(proc_obj.memory.json()),indent=4)

fp = open('/Users/ryanmukai/Documents/github/langchain_1/other/memory.json','w')

fp.write(json_string)

fp.close()
