from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid,io
from PIL import Image, ImageDraw, ImageFont
from matplotlib.pyplot import imshow
import requests,json
import base64
from openai import AzureOpenAI
import datetime
import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.ai.formrecognizer import DocumentAnalysisClient
import numpy as np
import pandas as pd
import re
from msg_parser import MsOxMessage
import streamlit as st
from io import BytesIO

def find_last_digit(s): # 20240605
    for i in range (len(s)-1,-1,-1):
        if s[i].isdigit():
            return i
    return -1

def find_first_digit(s): # 20240605
    for i in range (len(s)):
        if s[i].isdigit():
            return i
    return -1
 
def get_money_string(s):
    #match_st = re.search(r'\d', s)  #查找第一个数字
    #start = match_st.start()  # 获取数字的开始位置
    #match_ed = re.search(r'\d+$', s) #查找最后一个数字,对于'0.00'输出是None
    #end = match_ed.start()  # 获取数字的开始位置
    subs = s
    subs = subs.replace(" ", "") #去除掉字符串里的空格
    start = find_first_digit(subs)
    end = find_last_digit(subs)
    #print(s)
    if start!=-1 and end !=-1 and start<=end:        
        #print("st:ed",start,end+1)
        return subs[start:end+1] # 截取第一个非零数字及其后面的数字子串   
    else:
        return subs 
        
# attachments classification
# <snippet_creds_auth> train model has been done from custom vision UI
def attach_classify(image_path,blob_size): # 用MMG的模型
    startTime_pdf2img = datetime.datetime.now() #记录开始时间
    PRED_ENDPOINT = "https://customvisionmmg-prediction.cognitiveservices.azure.com/"
    prediction_key = "eed84aa785164f3da6574ad69a8ea6fe" #os.environ["VISION_PREDICTION_KEY"]
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(PRED_ENDPOINT, prediction_credentials)

    publish_iteration_name = "attach-classify-MMG-20240607" #用General A2做训练
    img_url = st.session_state['blob_client'].get_blob_sas(image_path)
 
    project_id ="0681bbcf-5898-4717-abf3-cfc9e1491c53"   # Project Id from custom vision UI settings
    #判断图片大小，不能大于4M
    if blob_size < 4*1024*1024 :
        results = predictor.classify_image_url(project_id, publish_iteration_name, img_url) 
    else:
        blob_client = st.session_state['blob_client'].get_file(image_path)
        img_this = blob_client.download_blob()
        image_data = Image.open(img_this)
        scale = 0.8
        new_width = int(image_data.width * (1 / scale))
        new_height = int(image_data.height * (1 / scale))
        resized_image = image_data.resize((new_width, new_height))
        img_io_bytes = io.BytesIO()
        resized_image = resized_image.convert("RGB")
        resized_image.save(img_io_bytes,format = "JPEG")
        img_bytes = img_io_bytes.getvalue()
        results = predictor.classify_image(project_id, publish_iteration_name, img_bytes)

    endTime_pdf2img = datetime.datetime.now() #结束时间
    #print('image classification time = ',(endTime_pdf2img-startTime_pdf2img).seconds)
    return results.predictions[0]
    
def pdf2image(page,file_path,output_folder,i):   
    # 一页的pdf 转 png图像, jpg也是支持的，只需要把名字改成.jpg结尾就行
    startTime_pdf2img = datetime.datetime.now() #记录开始时间
    tmp,file_name = file_path.split('/')
    imagePath = output_folder + '/'+ file_name[:-4] + '_'+ str(i) + '.png'
    print(imagePath)
    rotate = int(0)
    #每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6倍的图像
    #此处若是不做设置，默认图片大小为：792*612，dpf=96
    zoom_x = 1.33333333 # (1.33333333 --> 1056*816) (2 --> 1584*1224)
    zoom_y = 1.33333333
    mat = fitz.Matrix(zoom_x,zoom_y).prerotate(rotate)
    pix = page.get_pixmap(matrix=mat,alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    bytesIO = BytesIO()
    img.save(bytesIO,format='PNG')
    st.session_state['blob_client'].upload_image(bytesIO.getvalue(), imagePath) 
    endTime_pdf2img = datetime.datetime.now() #结束时间
    #print('pdf2img时间=',(endTime_pdf2img-startTime_pdf2img).seconds)
    return imagePath

def invoice_keywords_judge(content, keywords): #用MMG的AOAI模型
    api_base = 'https://mmgaustraliaeastopenai.openai.azure.com/' 
    api_key = '3c7360df29ba4a96834d2f7842e7734d'
    deployment_name = 'GPT-4-1106-Preview'
    api_version = '2024-02-01'

    client = AzureOpenAI(
        azure_endpoint=api_base,
        api_key=api_key,  
        api_version=api_version
    )
    #print(content)
    inputcontent = content + " \n 上面这段话中，有 " + keywords + " 这几个字么？有的话输出Yes，没有的话输出No"    
    #print(inputcontent)

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": inputcontent,
            },
        ],
        temperature=0.7,
        top_p=0.95,
        max_tokens=2000
    )

    print(response.choices[0].message.content)  
    return response.choices[0].message.content

def approve_words_judge(content, keywords): #用MMG的AOAI模型
    api_base = 'https://mmgaustraliaeastopenai.openai.azure.com/' 
    api_key = '3c7360df29ba4a96834d2f7842e7734d'
    deployment_name = 'GPT-4-1106-Preview'
    api_version = '2024-02-01'

    client = AzureOpenAI(
        azure_endpoint=api_base,
        api_key=api_key,  
        api_version=api_version
    )

    inputcontent = content + " \n In the paragraph above, does it contain the following words: " + keywords + " ？and at the meantime, does it contain a person who is the approver? If both answers are yes, output yes, elsewise output no."    
    #print(inputcontent)

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": inputcontent,
            },
        ],
        temperature=0.7,
        top_p=0.95,
        max_tokens=2000
    )

    #print(response.choices[0].message.content)  
    return response.choices[0].message.content

def approve_GPT4vision_enhance_judge(image_path, keywords): #用MMG的AOAI模型和CV模型
    # taxi receipt key words extraction by GPT4-vision with cv enhancement 
    startTime_pdf2img = datetime.datetime.now() #记录开始时间
    #print(image_path)
    blob_client = st.session_state['blob_client'].get_file(image_path)
    img_this = blob_client.download_blob()
    image_data = Image.open(img_this) 
    img_io_bytes = io.BytesIO()
    image_data.save(img_io_bytes,format = "PNG")
    img_bytes = img_io_bytes.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode('ascii')

    aoai_api_base = 'https://mmgaustraliaeastopenai.openai.azure.com/' 
    gpt4v_deployment_name = 'GPT-4-Vision-Preview'
    aoai_API_KEY = '3c7360df29ba4a96834d2f7842e7734d'

    base_url = f"{aoai_api_base}/openai/deployments/{gpt4v_deployment_name}" 
    headers = {   
        "Content-Type": "application/json",   
        "api-key": aoai_API_KEY 
    } 

    inputcontent = keywords + " If the answer is Yes, output yes, elsewise output no."
    # Prepare endpoint, headers, and request body 
    endpoint = f"{base_url}/extensions/chat/completions?api-version=2023-12-01-preview" 
    data = {
        "model": gpt4v_deployment_name,
        "enhancements": {
            "ocr": {
              "enabled": True
            },
            "grounding": {
              "enabled": True
            }
        },
        "dataSources": [
        {
            "type": "AzureComputerVision",
            "parameters": {
                "endpoint": "https://mmgaustraliaeastcv.cognitiveservices.azure.com/",
                "key": "6bc32ae82d014110a0394fbd578630f8"
            }
        }],
        "messages": [       
            { "role": "user", 
              "content": [
                {
                    "type": "text",
                    "text": inputcontent
                },
                { 
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                }]
            } 
        ], 
        "temperature": 0.01,
        "top_p": 0.95,
        "max_tokens": 2000 
    } 
    
       # Make the API call   
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))   

    endTime_pdf2img = datetime.datetime.now() #结束时间
    print('approve function GPT4-vision with cv enhancement post time=',(endTime_pdf2img-startTime_pdf2img).seconds)
    #print(f"Status Code: {response.status_code}")   
    result = response.json()
    #print(result['choices'][0]['message']['content'])
    return result['choices'][0]['message']['content']

def chi_invoice_extract_intell(file_path): # 输入的是这页的内容，输出的是提取出来的时间，类别，金额和币种  #用MMG的自定义表单识别模型
    #这里有个假设条件是，一页内容里已经包含了要找的一条信息的所有了。
    startTime_pdf2img = datetime.datetime.now() #记录开始时间
    endpoint = "https://mmgreceiptcheck.cognitiveservices.azure.com/"
    key = "631c348910884330a4c5caa1a349ab34"

    #model_id = "prebuilt-layout"
    #model_id = "prebuilt-invoice"
    model_id = "Chi-Invoice-MMG-20240607" # customized model named as Chi-Invoice with 14 invoices and 8 key words, include medical invoice
    document_analysis_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    blob_url = st.session_state['blob_client'].get_blob_sas(file_path)
    poller = document_analysis_client.begin_analyze_document(model_id, AnalyzeDocumentRequest(url_source=blob_url))
    result = poller.result()
    endTime_pdf2img = datetime.datetime.now() #结束时间
    #print('Document Intelligence new SDK use time=',(endTime_pdf2img-startTime_pdf2img).seconds)
    #print(result['documents'][0]['fields'])
    #print('发票号码 ', result['documents'][0]['fields']['invoice-number']['content'])
    #print('日期 ', result['documents'][0]['fields']['Data']['content'])
    #print('价税合计 ', result['documents'][0]['fields']['total-money']['content'])
    #print('销售方名称 ', result['documents'][0]['fields']['seller']['content'])
    #print('不计税金额 ', result['documents'][0]['fields']['pure-money']['content'])
    #print('税额 ', result['documents'][0]['fields']['tax-money']['content'])
    keywords = result['documents'][0]['fields']
    #print(keywords)
    #print(result['content'])
    attach_date = attach_type = attach_amount = attach_currency = attach_yes_invoice = '' #下面的每一个都应该先判断一下存在，再赋值
    attach_date = keywords['date']['content']
    if 'content' in keywords['total-money']:
        attach_amount = get_money_string(keywords['total-money']['content'])
    else:
        print('total-money is not detected!')
        pure_money = get_money_string(keywords['pure-money']['content'])
        tax_money = get_money_string(keywords['tax-money']['content'])
        attach_amount = str(float(pure_money) + float(tax_money))
    attach_currency = 'CNY' #定额增值税发票默认只有中国有，发票上都是以人民币币种结算的 
    if 'content' in keywords['service-content']:
        print('服务内容 ', keywords['service-content']['content'])
        if '住宿' in keywords['service-content']['content'] or '房屋租金' in keywords['service-content']['content']:
            attach_type = 'Accommodation(DOM)'
        elif '机票' in keywords['service-content']['content'] or '航' in keywords['service-content']['content']:
            attach_type = 'Airfares (DOM)'
        elif '运输' in keywords['service-content']['content']:
            attach_type = 'Taxi/CarHire/Incidentals (DOM)'
        elif '餐' in keywords['service-content']['content']:
            attach_type = 'Meals'
        elif '物流' in keywords['service-content']['content'] or '图书' in keywords['service-content']['content'] or '纸' in keywords['service-content']['content'] or '签证' in keywords['service-content']['content'] or '护照' in keywords['service-content']['content']: 
            attach_type = 'Office & Admin Supplies'
        elif '电信' in keywords['service-content']['content']:
            attach_type = 'Telephone Subsidies (Business)'
        elif '公证' in keywords['service-content']['content'] or '信息技术' in keywords['service-content']['content']:
            attach_type = 'Licences and Permits'
        elif '医疗' in keywords['service-content']['content'] or '药' in keywords['service-content']['content'] or '诊疗' in keywords['service-content']['content']:
            attach_type = 'Travel Health & Medicals'
    
    if 'content' in keywords['title']:
        print(keywords['title']['content'])
        if '普通' in keywords['title']['content']:
            attach_yes_invoice = 'Yes' 
        elif '专用' in keywords['title']['content'] or '用发票' in keywords['title']['content']:
            keywords_res = invoice_keywords_judge(str(result['content']),"抵扣联") # 抵扣联三个字有可能识别的不准确，例如：扣识别成和
            if 'Yes' in keywords_res:
                attach_yes_invoice = 'No'
            else:
                attach_yes_invoice = 'Yes'
        else:
            print("type of invoice is missed!")
    return  attach_date, attach_type, attach_amount, attach_currency, attach_yes_invoice

def receipts_detection(image_path,blob_size,output_folder,save_name): #需要发票是竖着可读的方向才行，横着那种不行
    # train and taxi receipts detection,including international meal receipts (save cut region and class)
    # <snippet_creds> train model has been done from custom vision UI
    PRED_ENDPOINT = "https://australiseastcustomvision-prediction.cognitiveservices.azure.com/"
    prediction_key = "a2ad6e09d1624fe08572a2ace482ee3e" #os.environ["VISION_PREDICTION_KEY"]
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(PRED_ENDPOINT, prediction_credentials)
    #publish_iteration_name = "taxi-train-Iteration4"
    #publish_iteration_name = "detection-Iteration7" #增加了两张高速费的样本做目标检测
    #publish_iteration_name = "detection-Iteration8" #增加了两张打车小票的样本做目标检测 0312温良
    #publish_iteration_name = "detection-Iteration9" #增加了打车票和高速费的样本做目标检测1212章箐Oct 23 - Nov 22
    publish_iteration_name = "detection-Iteration10" #增加了两张高速费做目标检测样本0202温良
    project_id ="8203d8be-dc10-482d-875d-a3e2520cb599"   # Project Id from custom vision UI settings
    print(image_path) 
    tmp, file_name = image_path.split('/')
    img_url = st.session_state['blob_client'].get_blob_sas(image_path)

    blob_client = st.session_state['blob_client'].get_file(image_path)
    img_this = blob_client.download_blob()
    image_data = Image.open(img_this)   
    width, height = image_data.size    
    #判断图片大小，不能大于4M 20240606
    if blob_size < 4*1024*1024 :
        results = predictor.detect_image_url(project_id, publish_iteration_name, img_url)  
    else:
        scale = 0.8
        new_width = int(image_data.width * (1 / scale))
        new_height = int(image_data.height * (1 / scale))
        resized_image = image_data.resize((new_width, new_height))
        img_io_bytes = io.BytesIO()
        resized_image = resized_image.convert("RGB")
        resized_image.save(img_io_bytes,format = "JPEG")
        img_bytes = img_io_bytes.getvalue()
        results = predictor.detect_image(project_id, publish_iteration_name, img_bytes)
        
    # Save the results.    
    det_threshold = 0.6
    subimg_list = []
    if results is not None:
        index = 1
        for prediction in results.predictions:
            #print("i,score: ",index,prediction.probability)
            if prediction.probability > det_threshold:
                #print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
                x = int(prediction.bounding_box.left*width+0.5)
                y = int(prediction.bounding_box.top*height+0.5)
                w = int(prediction.bounding_box.width*width+0.5)
                h = int(prediction.bounding_box.height*height+0.5)
                subimg = image_data.crop((x, y, x+w, y+h))
                subimg_path = output_folder + "/"+ save_name + "_" + prediction.tag_name + "_" + str(index) + '.png'#image_path[-4:]
                subimg_list.append([subimg_path,x,y,w,h])
                index = index + 1
                print(subimg_path)
                print(x,y,w,h)
                bytesIO = BytesIO()
                subimg.save(bytesIO,format='PNG')
                st.session_state['blob_client'].upload_image(bytesIO.getvalue(), subimg_path) 
        if len(subimg_list)==0: #如果上面没有找到的话，那就是阈值比较高，此时既然已分类成小票类型，那么至少返回一个最大的结果作为目标检测的结果    20240616
            prediction = results.predictions[0]
            x = int(prediction.bounding_box.left*width+0.5)
            y = int(prediction.bounding_box.top*height+0.5)
            w = int(prediction.bounding_box.width*width+0.5)
            h = int(prediction.bounding_box.height*height+0.5)
            subimg = image_data.crop((x, y, x+w, y+h))
            subimg_path = output_folder + "/"+ file_name[:-4] + "_" + prediction.tag_name + "_" + str(index) + '.png'#image_path[-4:]
            subimg_list.append([subimg_path,x,y,w,h])
            print(subimg_path)
            print(x,y,w,h)
            bytesIO = BytesIO()
            subimg.save(bytesIO,format='PNG')
            st.session_state['blob_client'].upload_image(bytesIO.getvalue(), subimg_path) 
            
    #test_data.close()
    print(file_name, 'sub image detection done!')
    return subimg_list

def keywords_extract_GPT4vision_enhance(image_path): #用MMG的AOAI模型和CV模型
    # taxi receipt key words extraction by GPT4-vision with cv enhancement 
    startTime_pdf2img = datetime.datetime.now() #记录开始时间
    #print(image_path)
    blob_client = st.session_state['blob_client'].get_file(image_path)
    img_this = blob_client.download_blob()
    image_data = Image.open(img_this) 
    img_io_bytes = io.BytesIO()
    image_data.save(img_io_bytes,format = "PNG")
    img_bytes = img_io_bytes.getvalue()
    encoded_image = base64.b64encode(img_bytes).decode('ascii')

    aoai_api_base = 'https://mmgaustraliaeastopenai.openai.azure.com/' 
    gpt4v_deployment_name = 'GPT-4-Vision-Preview'
    aoai_API_KEY = '3c7360df29ba4a96834d2f7842e7734d'

    base_url = f"{aoai_api_base}/openai/deployments/{gpt4v_deployment_name}" 
    headers = {   
        "Content-Type": "application/json",   
        "api-key": aoai_API_KEY 
    } 

    # Prepare endpoint, headers, and request body 
    endpoint = f"{base_url}/extensions/chat/completions?api-version=2023-12-01-preview" 
    data = {
        "model": gpt4v_deployment_name,
        "enhancements": {
            "ocr": {
              "enabled": True
            },
            "grounding": {
              "enabled": True
            }
        },
        "dataSources": [
        {
            "type": "AzureComputerVision",
            "parameters": {
                "endpoint": "https://mmgaustraliaeastcv.cognitiveservices.azure.com/",
                "key": "c48dfaaad11f4dbfaf746a8c1bd8ca6f"
            }
        }],
        "messages": [       
            { "role": "user", 
              "content": [
                {
                    "type": "text",
                    "text": "Find the date, currency and the total amount of money charged in the image, \
                    and determine its category from the following categories: \
                    meal, accommodation, airfares, taxi ticket, highway fees, train tickets, subscription fee, membership fee, medical, telephone, \
                    and output it in the format of 'Money': ;'Currency': ; 'Date': ;'Type': . \
                    Following the rules below:  \
                    1. Be based on facts, don't make up stories. \
                    2. If the output content can't been found, it should be outputed as blank. \
                    3. If it contains food, it should be identified as the meal type. \
                    4. If it contains mobile or wifi, it should be identified as telephone type. \
                    5. Determine the currency based on the content in the image. \
                    6. If it is a taxi ticket, the total amount of money charged is the total amount or Chinese called 实收金额, not the fare amount or Chinese called 金额. \
                    "
                    #4. Find the largest money and if it is larger than the total amount charged, the output Money should be the largest one.
                },
                { 
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                }]
            } 
        ], 
        "temperature": 0.01,
        "top_p": 0.95,
        "max_tokens": 2000 
    }   

    # Make the API call   
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))   

    endTime_pdf2img = datetime.datetime.now() #结束时间
    print('GPT4-vision with cv enhancement post time=',(endTime_pdf2img-startTime_pdf2img).seconds)
    #print(f"Status Code: {response.status_code}")   
    result = response.json()
    print(result['choices'][0]['message']['content'])

    attach_date = attach_type = attach_amount = attach_currency = ''
    thislist = result['choices'][0]['message']['content'].split(';')
    print(thislist)
    if len(thislist)>=4: # 找到了对应的各项  20240616
        thismoney = thislist[0]
        thiscurrency = thislist[1]
        thisdate = thislist[2] 
        thistype = thislist[3]
        #print(thismoney, thiscurrency, thisdate, thistype)
    else: #没找全怎么办，需要继续处理 20240616
        return  attach_date,attach_type,attach_amount,attach_currency
    
    if 'Currency' in thiscurrency:
        tmp,attach_currency = thiscurrency.split(':')
    domorint = '(DOM)'       
    if 'CNY' in attach_currency or 'yuan' in attach_currency or 'Yuan' in attach_currency or '元' in attach_currency or '¥' in attach_currency or 'RMB' in attach_currency:
        attach_currency = 'CNY'
        domorint = '(DOM)'
    else:
        domorint = '(INT)'
    
    if 'Type' in thistype:
        tmp,attach_type = thistype.split(':')
    if 'Taxi' in thistype or 'taxi' in thistype or 'train' in thistype or 'Train' in thistype or 'highway' in thistype or 'Highway' in thistype:
        attach_type = 'Taxi/CarHire/Incidentals' + domorint
    elif 'accommodation' in thistype or 'Accommodation' in thistype:
        attach_type = 'Accommodation' + domorint
    elif 'airfares' in thistype or 'Airfares' in thistype:
        attach_type = 'Airfares' + domorint
    elif 'Subscription' in thistype or 'subscription' in thistype:
        attach_type = 'Training-courses, Training, Seminar, Conferences'
    elif 'Membership' in thistype or 'membership' in thistype:
        attach_type = 'Employee Memberships/Subscript'
    elif 'Medical' in thistype or 'medical' in thistype:
        attach_type = 'Travel Health & Medicals'
    elif 'telephone' in thistype or 'Telephone' in thistype:
        attach_type = 'Telephone Subsidies (Business)'       
        
    if 'Date' in thisdate:
        pos = thisdate.index(':')+1 #因为Date里会混入小时分钟的表达里含有冒号，所以作此修改   20240605
        attach_date = thisdate[pos:]
        #tmp,attach_date = thisdate.split(':')
    if 'Money' in thismoney:
        tmp,attach_amount = thismoney.split(':')
        attach_amount = get_money_string(attach_amount)

    #print(attach_date,attach_type,attach_amount,attach_currency)
    return  attach_date,attach_type,attach_amount,attach_currency 

def get_image_area(image_path):
    blob_client = st.session_state['blob_client'].get_file(image_path)
    img_this = blob_client.download_blob()
    img = Image.open(img_this) 
    w,h = img.size	# 宽高像素
    imagePixmap = (float)(w*h/1000)
    #print("w,h:",w,h)
    #print("imagePixmap:", imagePixmap)
    return imagePixmap

def extract_one_attach_keywords(blob_name, blob_size,file_type, mres,temp_image_folder): # 从此处开始，所有用到的子函数都是为了云上的调用而修改
    #temp_image_folder = './temp_attach_images/'
    #file_type = "DOM"
    #file_type = "DOMINT"
    #file_type = "INT"
    approve_path = ""
    approve_flag = ""
    #mres = np.empty(shape=[0, 10], dtype=str) # 多加一个页数和一个页数上的item，也就是如果是pdf有多页,每页有多项的话，就会知道是第几页的第几项
    # only deal with pdf, msg and image type attchment
    file_Path = blob_name
    tmp, file_name = file_Path.split('/')
    print(file_Path)
    if file_name[-3:] == "jpg" or file_name[-3:] == "png": 
        cls = attach_classify(file_Path,blob_size) 
        print(cls)
        if cls.tag_name == "zzzzz" and cls.probability > 0.45: #需要用score限制一下，因为没有分other类别，如果是一条table的邮件回复啥的，score会低。
            result = [file_name,'1','1',cls.tag_name,str(cls.probability),'','','','','']
            result[5],result[6],result[7],result[8],result[9] = chi_invoice_extract_intell(file_Path) 
            print(result)
            mres = np.append(mres, [result], axis = 0) 
        elif cls.tag_name == "sssss" and cls.probability > 0.45:
            subimg_list = receipts_detection(file_Path,blob_size,temp_image_folder,file_name[:-4])
            if len(subimg_list) == 0:
                print("image do not detect receipts!!! need to detect again!")
                return mres,approve_flag,approve_path
            subimg_area = get_image_area(subimg_list[0][0]) 
            img_area = get_image_area(file_Path)
            print(subimg_area,0.7*img_area)
            if len(subimg_list) == 1 and file_type=="DOM" and subimg_area>0.7*img_area:
                #国内文件类型错分成了小票类型? 哪个样例？
                print("wrong image")
                return mres,approve_flag,approve_path
            #开始提取每一个出租车小票的关键词信息
            j=1
            for subimg in subimg_list:
                result = [file_name,'1',str(j),cls.tag_name,str(cls.probability),'','','','',''] 
                #print(subimg)
                result[5],result[6],result[7],result[8] = keywords_extract_GPT4vision_enhance(subimg[0])
                j=j+1
                print(result)
                mres = np.append(mres, [result], axis = 0) 
        elif cls.tag_name == "wwwww" and cls.probability > 0.45:
            if file_type=="DOM": #一张出租车小票或者高速票被分成了文件类型，确认一下
                #result[5],result[6],result[7],result[8] = keywords_extract_GPT4vision_enhance(file_Path)
                print("DOM & wwwww & image")
            else:  #国内的报销的文档类型的附件忽略，只处理国外的
                result = [file_name,'1','1',cls.tag_name,str(cls.probability),'','','','','']
                result[5],result[6],result[7],result[8] = keywords_extract_GPT4vision_enhance(file_Path)
                print(result)
                if result[8] !='CNY':#国内外混杂情况下，如果这一份附件是国内的，忽略，只处理国外的。这里默认Uber的都是在国外打车，而且币种识别正确
                    mres = np.append(mres, [result], axis = 0) 
        #暂时先不处理fffff类型，这个更多和付费细节相关，不确定是不是和币种金额转换有关。  
    elif file_name[-3:] == "pdf": 
        pdf_blob_client = st.session_state['blob_client'].get_file(file_Path)
        pdf_bytes = pdf_blob_client.download_blob().content_as_bytes()
        pdfDoc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(pdfDoc.page_count):
            image_path = pdf2image(pdfDoc[i],file_Path,temp_image_folder,i+1) #对每一页做pdf到图像的转换,并保存
            print(image_path)
            cls = attach_classify(image_path,blob_size)
            print(cls)
            if cls.tag_name == "zzzzz" and cls.probability > 0.45: #需要用score限制一下，因为没有分other类别，如果是一条table的邮件回复啥的，score会低。
                result = [file_name,str(i+1),'1',cls.tag_name,str(cls.probability),'','','','','']
                result[5],result[6],result[7],result[8],result[9] = chi_invoice_extract_intell(image_path) 
                print(result)
                mres = np.append(mres, [result], axis = 0) 
            elif cls.tag_name == "sssss" and cls.probability > 0.38: # 0118高辉晓-27-Expense Receipt_026的score分比较低，所以这里改成了0.38
                save_name = file_name[:-4]+ '_page_' + str(i+1)
                subimg_list = receipts_detection(image_path,blob_size,temp_image_folder,save_name)
                if len(subimg_list) == 0:
                    print("pdf do not detect receipts!!! need to detect again!")
                    return mres,approve_flag,approve_path
                subimg_area = get_image_area(subimg_list[0][0]) 
                img_area = get_image_area(image_path)
                print(subimg_area,0.7*img_area)
                if len(subimg_list) == 1 and file_type=="DOM" and subimg_area > 0.7*img_area:
                    #国内文件类型错分成了小票类型
                    print("wrong pdf")
                    return mres,approve_flag,approve_path
                j=1
                for subimg in subimg_list:
                    result = [file_name,str(i+1),str(j),cls.tag_name,str(cls.probability),'','','','','']
                    #print(subimg)
                    result[5],result[6],result[7],result[8] = keywords_extract_GPT4vision_enhance(subimg[0])
                    j=j+1
                    print(result)
                    mres = np.append(mres, [result], axis = 0) 
            elif cls.tag_name == "wwwww" and cls.probability > 0.45: 
                #page_text = pdfDoc[i].get_text("text")
                #print(page_text)
                #if approve_flag =="":
                    #output_flag = approve_GPT4vision_enhance_judge(image_path,"Does it is contain the words: Authorithy to Travel?")
                    #print(output_flag)
                    #if 'yes' or 'Yes' in output_flag:
                    #    third_image_path = pdf2image(pdfDoc[i+2],temp_image_folder,f[:-4],i+2+1) #对每一页做pdf到图像的转换
                    #    third_output_flag = approve_GPT4vision_enhance_judge(third_image_path,"Does it is contain a manager approver? ")
                    #    print(third_output_flag)
                    #    if 'yes' or 'Yes' in third_output_flag:                 
                    #        approve_flag = "Yes"
                    #        approve_path = file_Path
                    #        break
                if file_type!="DOM":#国内的报销的文档类型的附件忽略，只处理国外的  #国外的文档如果有多页的话，这里需要处理一下，还没处理！！！！！！
                    result = [file_name,str(i+1),'1',cls.tag_name,str(cls.probability),'','','','','']
                    result[5],result[6],result[7],result[8] = keywords_extract_GPT4vision_enhance(image_path)
                    print(result)
                    if result[8] =='' or result[8] ==' ':
                        break
                    if 'Taxi'in result[6] and result[8] !='CNY':#国内外混杂情况下，如果这一份附件是国内的，忽略，只处理国外的。这里默认Uber的都是在国外打车，而且币种识别正确
                        mres = np.append(mres, [result], axis = 0) 
                    if 'Medical' in result[6] or 'meal' in result[6]: #有中文水单做医疗体检的情况
                        mres = np.append(mres, [result], axis = 0) 
                    if 'Training' in result[6] or 'training' in result[6] or 'Subscript' in result[6] or 'Accommodation' in result[6]:
                        mres = np.append(mres, [result], axis = 0)
            #暂时先不处理fffff类型，这个更多和付费细节相关，不确定是不是和币种金额转换有关。
            elif cls.tag_name == "fffff" and cls.probability > 0.35: #0118高辉晓-12-Expense Receipt_011 的score分错了，还比较低，所以这里改成了0.35
                #有可能是中文医院定额发票错分成fffff类型了
                result = [file_name,str(i+1),'1',cls.tag_name,str(cls.probability),'','','','','']
                result[5],result[6],result[7],result[8] = keywords_extract_GPT4vision_enhance(image_path)
                if 'medical' in result[6] or 'Medical' in result[6]:
                    result[6] = 'Travel Health & Medicals'
                    mres = np.append(mres, [result], axis = 0)
                if 'Taxi' in result[6] or 'Accommodation' in result[6] or 'Telephone' in result[6]:
                    mres = np.append(mres, [result], axis = 0)
        pdfDoc.close()
    elif file_name[-3:] == "msg": 
        msg_blob_client = st.session_state['blob_client'].get_file(file_Path)
        msg_this = msg_blob_client.download_blob().content_as_bytes()
        msg_obj  = MsOxMessage(msg_this)
        msg_properties_dict = msg_obj.get_properties()
        #print(msg_properties_dict['Subject'])
        #print(msg_properties_dict['recipients'])
        #print(msg_properties_dict['Body'])
        output_flag = approve_words_judge(msg_properties_dict['Body'],"Authority to Travel request you submitted has been approved")
        print(output_flag)
        if 'yes' or 'Yes' in output_flag:
            approve_flag = "Yes"
            approve_path = file_Path
            
    return mres,approve_flag,approve_path
  
#20240615  
def find_same_items(df_extract_edit,df_extract_filenum,df_init_edit): #从主表里找到和确认提取的项里相同的项，这里的输入是除去了一些列的输入，df_extract_edit 除去了原有的data列，df_init_edit除去了原有的'Cost Centre WBS'列
    #df_extract_filenum是提取到关键词的附件的总个数，不是真正的总附件数，看看之后需不需要改。
    ext_list = []
    init_list = []
    #目前是n*n的复杂度，需要优化  20240616
    for i in range(df_extract_filenum):
        if df_extract_edit['Amount'].iloc[i]!=None:
            #print(df_extract_edit['Amount'].iloc[i])
            #print(df_init_edit['Receipt Amount'])
            value_to_find = float(df_extract_edit['Amount'].iloc[i])
            positions = []
            for j, v in enumerate(df_init_edit['Receipt Amount']):
                #print("value_to_find:",value_to_find)
                if not np.isreal(v):
                    v = v.replace(',', '')
                #print("i,v:",j,float(v))
                target_str = df_extract_edit['Crcy'].iloc[i].replace(' ', '')
                #print(target_str)
                source_str = df_init_edit['Crcy'].iloc[j].replace(' ', '')
                #print(source_str)
                if float(v)==value_to_find and target_str==source_str:
                   positions.append(j) 
            #positions = df_init_edit.index[df_init_edit['Receipt Amount'] == value_to_find].tolist()
            print(positions)
            if len(positions): #找到，并且有且仅有一个
                print(df_init_edit['Expense Type'].iloc[positions[0]])
                print(df_extract_edit['Type'].iloc[i])
                if 'Meal' in df_extract_edit['Type'].iloc[i]: # 如果找到对应的Meal项，用主表的具体的Meal的类型给提取到的Meal类型赋值。 df_extract_edit 让显示页面的结果直接改变了？
                    #print(df_init_edit.iloc[positions[0]][2])
                    df_extract_edit['Type'].iloc[i] = df_init_edit['Expense Type'].iloc[positions[0]]
                elif df_extract_edit['Type'].iloc[i][1:4] != df_init_edit['Expense Type'].iloc[positions[0]][1:4]: # 用type字符串的1到4个字母来判断两者是否相同   20240616
                    continue
                ext_list.append(i)
                init_list.append(positions[0]) #这里保留的是数组变量的位置，No.列里的数字需要这个值加上1
    print(ext_list)   
    print(init_list)
    return ext_list,init_list