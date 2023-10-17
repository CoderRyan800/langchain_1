import json

from base_code.pdf_processor import *

proc_obj = pdf_processor(pdf_filename_list=[
    "/home/ryanmukai/Documents/github/redesigned-octo-goggles/writing/lstm_paper.pdf",
    "/home/ryanmukai/Documents/github/redesigned-octo-goggles/writing/Dual Agent Demo.ipynb - Colaboratory.pdf",
    "/home/ryanmukai/Documents/github/langchain_1/writing/2107.08924.pdf",
    "/home/ryanmukai/Documents/github/langchain_1/writing/2211.08701v1.pdf"
    ],
    chat_gpt_model_choice='gpt-4')

query = "The input provided contains a draft paper on LSTM agents and self awareness, a demonstration of two LSTM-based agents performing a task requiring self-awareness, and two papers on epistemic self-knowledge in neural networks, in that order.  Our goal is to revise the first paper, especially in light of the following papers 'Epistemic Neural Networks' and 'Interpretable Self-Aware Neural Networks for Robust Trajectory Prediction', which also cover epistemic self-awareness.  Please first evaluate the LSTM agents paper and tell me whether it represents a publication-worthy innovation in light of the existing literature.  "
response = proc_obj.query_the_document(query)

fp = open('paper_innovation_evaluation.txt','w')

fp.write(response['answer'])

fp.close()

query = "Then please write a step-by-step analysis of the LSTM paper with an eye toward making it publication worthy in a peer-reviewed conference or journal such as PlosONE.  Be sure to analyze the LSTM paper in light of the two following epistemic self awareness papers."

response = proc_obj.query_the_document(query)

fp = open('paper_analysis.txt','w')

fp.write(response['answer'])

fp.close()

query = "Then please write an outline for an entirely new paper that could be published in a peer-reviewed journal such as PlosONE.  Clearly state what the section titles could be and summarize what each section should contain.  I want to cite the two epistemic self-awareness papers included after the LSTM paper and its demo."

response = proc_obj.query_the_document(query)

fp = open('paper_outline_new.txt','w')

fp.write(response['answer'])

fp.close()

query = "Please provide a starter abstract to aid me in getting started."

response = proc_obj.query_the_document(query)

fp = open('paper_suggestions_intro.txt','w')

fp.write(response['answer'])

fp.close()

query = "Please provide a starter introduction to aid me in getting started."

response = proc_obj.query_the_document(query)

fp = open('paper_suggestions_intro.txt','w')

fp.write(response['answer'])

fp.close()

json_string = json.dumps(json.loads(proc_obj.memory.json()),indent=4)

fp = open('memory.json','w')

fp.write(json_string)

fp.close()
