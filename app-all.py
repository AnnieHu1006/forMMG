import streamlit as st
from streamlit_pdf_reader import pdf_reader
from pathlib import Path
import fitz
import os
from io import BytesIO
from utilities.azureblobstorage import AzureBlobStorageClient
import pandas as pd

sslist = ["file_name","output_root_path"]
for ss in sslist:
    if ss not in st.session_state:
        st.session_state[ss] = ""

st.session_state["selected_df"]=pd.DataFrame()
st.session_state["thumbnail"]=[]
if len(st.session_state["thumbnail"])==0:
    st.session_state["thumbnail"]=[]
else:
    st.session_state["thumbnail"].clear()
    st.session_state["thumbnail"]=[]

# me 
#st.session_state['blob_account_name'] ='blobcanadagpt'
#st.session_state['blob_account_key'] ='HzyUsT16rzGYCr1k7mOolmQyXjv/rQw5yUSDlRK4g1wszJW8pE2cCYnP6LkSVN70tH+d/LYu0+4W+AStDtCTkw=='
#st.session_state['blob_container_name'] = 'mmg-materials'

# MMG Sujia
st.session_state['blob_account_name'] ='mmgreceiptmaterials'
st.session_state['blob_account_key'] ='JViTlzwJQcGKq3chpdYUcAGmmuamThnEbF3++dlpwrYJBv/GpQabr5fRAbp136PlFsWCAjhbDUwP+AStbYBazA=='
st.session_state['blob_container_name'] = 'mmg-meterials'

st.session_state['blob_client']: AzureBlobStorageClient = None
st.session_state['blob_client'] = AzureBlobStorageClient(st.session_state['blob_account_name'],st.session_state['blob_account_key'],st.session_state['blob_container_name']) if st.session_state['blob_client'] is None else st.session_state['blob_client']


def split_main_and_attachments(this_pdf):
    # 得到文件的名字
    main_file_name = this_pdf.name[:-4]
    if st.session_state["file_name"]=="":
        st.session_state["file_name"] = main_file_name
    elif st.session_state["file_name"]!= main_file_name:
        print(st.session_state["file_name"]+"have been loaded!!! Need to change!!!")
        st.session_state["file_name"] = main_file_name
    
    attach_save_folder = main_file_name + "/"
    #os.makedirs(attach_save_folder, exist_ok=True)
    # 打开PDF文档
    pdf_file = fitz.open(stream=this_pdf.read(), filetype="pdf")
    out_main_pdf = fitz.open()
    out_main_pdf.insert_pdf(pdf_file)
    main_pdf_stream = BytesIO() # 是否需要缓存保存使得每次只有一个存储空间被占用？   20240620
    out_main_pdf.save(main_pdf_stream)
    main_pdf_bytes = main_pdf_stream.getvalue()
    st.session_state['blob_client'].upload_file(main_pdf_bytes, main_file_name + "-main.pdf", content_type='application/pdf')
    #pages = pdf_file.page_count
    #count = pdf_file.embfile_count()

    #遍历PDF文件的附件的名字,获得附件内容，并保存
    num=1
    for item in pdf_file.embfile_names():
        item_name = pdf_file.embfile_info(item)["filename"]
        #默认附件个数小于100个，所以以两位数字补全，让它是按顺序排列
        if num <10:
            strnum = '0'+str(num)
        else:
            strnum = str(num)
        #判断文件名里是否有乱码，乱码一般以'?????.pdf'的形式展示，所以这里查找是否有?这个符号，返回的是找到的?的位置；如果没有，返回的是-1
        if item_name.find('?')!=-1:
            one_outfile =  attach_save_folder + main_file_name + '-' + strnum + '-RRRRR' + item_name[-4:]
            print(one_outfile)
        else:
            one_outfile =  attach_save_folder + main_file_name + '-' + strnum + '-' + item_name # 这里的数字在大于10之后，会对顺序产生误导，需要转化成前面填零的表达方式
            print(one_outfile)
        num=num+1
        fData = pdf_file.embfile_get(item)
        st.session_state['blob_client'].upload_file(fData,one_outfile)
        #with open(one_outfile, 'wb') as outfile: 
        #    outfile.write(fData) 
    print(main_file_name + " attachment split and save is done!") 

# 主程序
def main():
    st.header("MMG 财务报销 单据比对系统")
    # 根据分页显示不同的内容
    st.subheader("上传pdf原文件")
    #print(os.getcwd())
    #print(os.path.dirname(os.getcwd()))
    #dir_root = os.path.dirname(os.getcwd())
    #output_root_path = dir_root + '\outputs'
    #print(output_root_path)
    #if not os.path.exists(output_root_path):
    #    os.makedirs(output_root_path)
    #st.session_state["output_root_path"] = output_root_path
    file = st.file_uploader("选择待上传的PDF文件", type=['pdf'])
    if file is not None:        
        #folder_path = os.path.join(output_root_path,file.name[:-4])
        #print(folder_path)
        #print(st.session_state["file_name"])
        #print(file.name[:-4])
        filename = file.name[:-4]
        
        if st.session_state['blob_client'].check_folder_exist(filename):
            if st.session_state["file_name"]=="":
                st.session_state["file_name"] = file.name[:-4]
                #print(st.session_state["file_name"]+" the first time to give value!")
            elif st.session_state["file_name"]!= file.name[:-4]:
                st.session_state["file_name"] = file.name[:-4]
                #print(st.session_state["file_name"]+"have been loaded!!! Need to delete")
            print(file.name + "文件夹存在")
            st.text("文件的文件夹存在，文件夹里之前已完成主页和附件的split操作") 
        else:
            print(file.name+" will be splited main and attaches!")
            split_main_and_attachments(file)
            st.text("文件已上传，主页和附件的split操作完成") 
            
    
            
            
if __name__ == '__main__':
    main()           
            
