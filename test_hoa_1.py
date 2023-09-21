import json

from code.pdf_processor import *

# MAIN_PATH = "/home/ryanmukai/Documents/github/langchain_1/documents/"

MAIN_PATH = "C:/Users/ryan2/Documents/github/langchain_1/documents/"


input_pdf_list = [
        MAIN_PATH + "Bylaws.pdf",
        MAIN_PATH + "CC&Rs.pdf",
        MAIN_PATH + "Rules & Regs.pdf",
        MAIN_PATH + "Rules for Election.pdf",
        MAIN_PATH + "Rules for Voting.pdf",
        MAIN_PATH + "CID STATEMENT Hubbard Gardens 03-21-2018.pdf",
        MAIN_PATH + "SI-COMPLETE Hubbard Gardens 03-21-2018.pdf",
        MAIN_PATH + "SI-COMPLETE Hubbard Gardens 03-09-2016.pdf",
        MAIN_PATH + "Articles of Incorporation Hubbard Gardens May 19 1998.pdf",
        MAIN_PATH + "Hubbard Gardens Search - Business Entities California Secretary of State.pdf",
        MAIN_PATH + "Hubbard Gardens HOA policies and budget 2021.pdf"
    ]

output_pdf_list = [

        MAIN_PATH + "Bylaws2.txt",
        MAIN_PATH + "CC&Rs2.txt",
        MAIN_PATH + "Rules & Regs2.txt",
        MAIN_PATH + "Rules for Election2.txt",
        MAIN_PATH + "Rules for Voting2.txt",
        MAIN_PATH + "CID STATEMENT Hubbard Gardens 03-21-2018 2.txt",
        MAIN_PATH + "SI-COMPLETE Hubbard Gardens 03-21-2018 2.txt",
        MAIN_PATH + "SI-COMPLETE Hubbard Gardens 03-09-2016 2.txt",
        MAIN_PATH + "Articles of Incorporation Hubbard Gardens May 19 1998 2.txt",
        MAIN_PATH + "Hubbard Gardens Search - Business Entities California Secretary of State 2.txt",
        MAIN_PATH + "Hubbard Gardens HOA policies and budget 2021 2.txt"

]

N = len(input_pdf_list)

for index in range(N):

    current_input_filename = input_pdf_list[index]
    current_output_filename = output_pdf_list[index]
    extract_UnstructuredPDFLoader(current_input_filename, current_output_filename)

proc_obj = pdf_processor(pdf_filename_list=output_pdf_list,
    chat_gpt_model_choice='gpt-4',pdf_flag=False)

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

fp = open('/home/ryanmukai/Documents/github/langchain_1/other/response_data.json','w')

for query in question_list:
    response = proc_obj.query_the_document(query)
    output_string = repr(response)
    fp.write(output_string)
    fp.write("\n")

fp.close()

json_string = json.dumps(json.loads(proc_obj.memory.json()),indent=4)

fp = open('/home/ryanmukai/Documents/github/langchain_1/other/memory.json','w')

fp.write(json_string)

fp.close()
