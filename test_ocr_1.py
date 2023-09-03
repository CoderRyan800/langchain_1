import pytesseract
from pdf2image import convert_from_path

pages = convert_from_path('test.pdf', 500)
text = ''

for page in pages:
    text += pytesseract.image_to_string(page)

fp = open('test.txt','w')
fp.write(text)
fp.close()

