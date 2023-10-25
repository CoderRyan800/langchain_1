from base_code.graph_creator import *


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

output_list = create_graph_from_documents(list_of_docs)

with open('test_script.txt','w') as fp:

    for current_item in output_list:
        fp.write("%s\n" % (current_item,))


