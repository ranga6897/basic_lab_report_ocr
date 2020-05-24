from ness_functions import *

import pytesseract
from pdf2image import convert_from_path
import numpy as np
import pandas as pd
import re
import os



PDF_file = "EHR IV Dr. Anil Kumar Patra.pdf"    # pdf path


pages = convert_from_path(PDF_file, 500) 
image_counter = 0

for page in pages: 

    filename = "page_"+str(image_counter)+".JPEG"
    page.save(filename, 'JPEG') # Save the image of the page in system 
    image_counter = image_counter + 1

    


for i in range(0, image_counter+1): 
        
    filename = "page_"+str(i)+".JPEG"
    raw_data = process(filename) #func    
    pattern_char = re.compile(r'\w*[./]*\w*')
    data_all = remove(raw_data,pattern_char)  #func
 

    report_type,test = find_report_type(data_all) #func

    if report_type == 'a':
        params_raw = params(data_all)   #func
        modified_params = params_modify(params_raw)  #func
        data_frame = create_data_frame(raw_data,modified_params,test)  #func
        
        #writing data into otherformats
        data_frame.to_csv("page_"+str(i)+".csv")
    #     data_frame.to_json("page_"+str(i)+".json") # uncomment to process
        
        # entire Text of image
        # f = open("page_"+str(i)+"_text"+".txt", "a")  # uncomment to process
        # if data_all is not None:
        #     for line in data_all:
        #         f.write(line)
        #         f.write('\n')
        # f.close()

    
    