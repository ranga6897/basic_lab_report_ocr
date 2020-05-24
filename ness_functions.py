# Import libraries 
from PIL import Image 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/ranga/AppData/Local/Tesseract-OCR/tesseract.exe'

import cv2
import spacy
from spacy.matcher import Matcher
import numpy as np
import pandas as pd
import re

def process(filename):       

    img = cv2.imread(filename,0)
    # img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # img = cv2.fastNlMeansDenoising(img)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,81,50)

    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # # img = cv2.erode(img, kernel, iterations=1)
    # new_image = 'edited' + '_' + image  
    # cv2.imwrite(new_image, img)


    read = pytesseract.image_to_string(img, lang = 'eng')
    # data = pytesseract.image_to_data(new_image, output_type='data.frame')  

    return read

def remove(data, pattern_re):
    raw_data = data
    pattern = pattern_re
    
    import re
    
    temp = []
    for line in raw_data.splitlines():
        matches = pattern.findall(line)
        temp_filter = list(filter(None, matches))
        temp_filter = ' '.join(temp_filter)
        temp.append(temp_filter)
        
    temp = list(filter(None, temp))
    if len(temp) > 0:
        return list(temp)

def params(data):   
    
    x = []
    import spacy
    from spacy.matcher import Matcher

    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)

    if data is not None:
        for line in data:
        #     pattern = [{'IS_ALPHA':True},{'IS_ALPHA':True},{'IS_ALPHA':True},{'IS_ALPHA':True},{'IS_ALPHA':True},{'LIKE_NUM':True},{'LIKE_NUM':True}]
            pattern = [{"IS_ALPHA": {"REGEX": "[a-zA-Z]*[.: ]*[a-zA-Z]*"}},{"IS_ALPHA": {"REGEX": "[a-zA-Z]*[.: ]*[a-zA-Z]*"}},{'LIKE_NUM':{"REGEX": "[0-9]*[.-]*[0-9]*"}}]
            matcher.add(line, None, pattern)

            doc = nlp(line)

            matches = matcher(doc)
            for match_id, start, end in matches:
                string_id = nlp.vocab.strings[match_id]  # Get string representation
                span = doc[start:end]  # The matched span
        #         span = doc[:3]
    #             print(span.text)
                x.append(span.text)
                break
    return x

def params_modify(params_raw):
    
    raw = params_raw

    modified_params = []        
    pattern = re.compile(r'[a-zA-Z0-9 ./]*\d$')
    for i in raw:
        matches = pattern.findall(i)
        modified_params.append(matches)
    modified_params = list(filter(None, modified_params))
    
    return modified_params


def find(data, pattern_re):
    raw_data = data
    pattern = pattern_re
    
    import re
    
    temp = []
    for line in raw_data.splitlines():
        matches = pattern.findall(line)
        temp.append(matches)
        
    temp_filter = list(filter(None, temp)) 
    if len(temp_filter) > 0:
        return np.array(temp_filter).ravel()

def find_all(data_raw):
    doc_pattern = re.compile(r'Dr\.?.?[a-z]+.?[a-z]*.?[a-z]*',re.I)
    date_pattern = re.compile(r'\d\d[/.]\d\d[/.]\d{4}')
    test_pattern = re.compile(r'[a-z]+[ .-][a-z]*[ .-]REPORT$',re.I)

    date = find(data_raw,date_pattern) #func
    test = find(data_raw,test_pattern) #func
    doctor = find(data_raw,doc_pattern) #func
    
#     if len(date) > 1 and len(date) < 3:
#         if date[0] == date[1]:
#             date.pop()
            
    
    return date,test,doctor

def test_search(data_,common_words_):
    raw = data_
    common_words = common_words_
    
    def clean(line):
        for word in common_words:
            if word.lower() in line.lower():
                return line
    
    filtered = list(filter(clean, raw))
  
    tests = []

    for line in raw:
        if line  in filtered:
            tests.append(line)
    
    if len(tests)>0:
        return True,tests
    else:
        return False,tests

def find_report_type(data):
    common_words_lab_report = common_words = ['CBC TEST','CBP','complete blood picture','bio chemistry','complete blood count','human immunodeficiency virus','hbsag rapid','hepatitis',
                       'hbaic','liver test','liver function test','lipid','lipid profile','serum','urine','urine examination','plasma','blood','glucose','sugar']
    common_words_radiology_report = ['Radiology', 'Computerized Tomography','MRI', 'Magnetic Resonance Imaging',
                            'MRA', 'Magnetic Resonance Angiography','Ultrasound' ,'radiologist','abdomen','pelivs']
    common_words_discharge = ['discharge summary','discharge']
    common_words_procedures = ['ECHOCARDIOGRAPHY','EGD','Holter','Colonoscopy']
    
    is_lab,lab_test = test_search(data, common_words_lab_report)
    is_radiology,radiology_test = test_search(data, common_words_radiology_report)
    is_discharge,discharge = test_search(data, common_words_discharge)
    is_procedure,procedure = test_search(data, common_words_procedures)
    
    if is_lab and not is_radiology and not is_discharge and not is_procedure:
        return 'a',lab_test
    elif is_radiology and not is_lab and not is_discharge and not is_procedure:
        return 'b',radiology_test
    elif is_discharge and not is_lab and not is_radiology and not is_procedure:
        return 'c',discharge
    elif is_procedure and not is_lab and not is_radiology and not is_discharge:
        return 'd',procedure

def create_data_frame(raw_data_,modified_params_,test):
    raw_data = raw_data_
    modified_params = modified_params_
    
    parameter = []
    value = []
    
    date,test_,doctor = find_all(raw_data)  #func, inp
    parameter.append('Date')
    value.append(date)
    parameter.append('Test_type')
    value.append(test)
    parameter.append('Doctor')
    value.append(doctor)
    
    def Re_alpha(s):
        return re.match(r"\.?[a-zA-Z]\.?", s) is not None
    def Re_numeric(s):
        return re.match(r"\.?[0-9]", s) is not None



    for i in np.array(modified_params).ravel(): #inp
        line = i.split()

        if  len(line)>2 and Re_alpha(line[0]) and Re_alpha(line[1]) and Re_numeric(line[2]):
            line[0] = line[0]+'_'+line[1]
            line[1] = line[2]

        elif len(line)>3 and Re_alpha(line[0]) and Re_alpha(line[1]) and Re_alpha(line[2]) and Re_numeric(line[3]):
            line[0] = line[0]+'_'+line[1]+'_'+line[2]
            line[1] = line[3]

        if len(line)>1:
            parameter.append(line[0])
            value.append(line[1])
#         parameter.append(line[0])
#         value.append(line[1])

#     print(parameter, value)
    
    df = pd.DataFrame(data = [parameter,value]).T
    return  df


