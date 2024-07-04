import streamlit as st
import tempfile
import base64
import datetime
import numpy as np
import pandas as pd
import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

sslist = ["file_type"]
for ss in sslist:
    if ss not in st.session_state:
        st.session_state[ss] = ""
if 'attach_table_type' not in st.session_state:
    st.session_state.attach_table_type = pd.DataFrame(columns=['file_name', 'page_num','item_num','cls', 'score','Date','Type','Amount','Crcy','Accept_Invoice'])

@st.cache_data
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data

@st.cache_data
def corridors():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        fp = Path(tmp_file.name)
        print(fp)
        fp.write_bytes(uploaded_file.getvalue())
        with open(tmp_file.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" ' \
                  f'width="800" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def contains_digit(s):
    return any(char.isdigit() for char in s)

def extract_main_file_table(file_path):
    # to extract pdf main table, receipt table, use new SDK
    startTime_pdf2img = datetime.datetime.now() #记录开始时间
    endpoint = "https://docintelnew.cognitiveservices.azure.com/"
    key = "04045862a7494346a2e8016e5b7efdbc"

    blob_url = st.session_state['blob_client'].get_blob_sas(file_path)
    #pdf_bytes = blob_client.download_blob().content_as_stream()
    # with new SDK content内容比 old SDK 更精确
    document_analysis_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    #with open(pdf_bytes, "rb") as f:
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-layout", AnalyzeDocumentRequest(url_source=blob_url))
    result = poller.result()

    colnum = 8  # from fixed format
    mres = np.empty(shape=[0, colnum], dtype=str)
    CNYnum = 0
    file_type = "" # DOM stand for internal China only, INT stand for external China only and DOMINT stand for both internal China and external China
    for table_idx, table in enumerate(result.tables):
        if "Receipt" in table.cells[0].content and "Additional" not in table.cells[0].content and table.column_count == colnum:
            for i in range(table.row_count):
                result = [''] * table.column_count
                for j in range(table.column_count):
                    result[j] = table.cells[i*table.column_count+j].content
                    #print(i,j,result[j])
                    if result[j] == '': #Cost Centre WBS 没有提取出来
                        result[j-1], result[j] = result[j-1].split('2')
                        result[j] = '2'+ result[j]
                if "Receipt" in result[0] and mres.size >0:
                    continue
                else:
                    mres = np.append(mres, [result], axis = 0)   
                    
                if result[table.column_count-1]=="CNY":
                    CNYnum = CNYnum+1
        else:
            continue
    if mres.shape[0]>1 and 'Date' not in mres[0][0] and 'Date' in mres[1][0]: #将主表的标题识别成了2行，需要合并    20240616
        for i in range(8):
            mres[0][i] = mres[0][i] + '\n' + mres[1][i]
        #删除第二行
        t_mres = mres.copy()
        r_mres = np.delete(t_mres,1,axis=0)
    else:
        r_mres = mres.copy()

    n,m = r_mres.shape
    #print(n,CNYnum)
    if CNYnum == n-1:
        file_type = "DOM"
    elif CNYnum == 0:
        file_type = "INT"
    else:
        file_type = "DOMINT"
 
    #f.close() 
    endTime_pdf2img = datetime.datetime.now() #结束时间
    print('extract main file table time =',(endTime_pdf2img-startTime_pdf2img).seconds)    
    #print(mres)
    return r_mres,file_type

def get_file_type(main_table):
    print(main_table)
    CNYnum = 0
    row_num,col_num = main_table.shape
    print("row number: " + str(row_num)) 
    print("column number: " + str(col_num))
    for index, row in main_table.iterrows():
        if row['Crcy'] == "CNY":
            CNYnum = CNYnum+1
            
    if CNYnum == row_num:
        return "DOM"
    elif CNYnum == 0:
        return "INT"
    else:
        return "DOMINT" 

def main():
    st.subheader("报销文件的主页面信息如下：")
    main_file_name = st.session_state["file_name"]
    file_path = main_file_name + "-main.pdf" 
    blob_client = st.session_state['blob_client'].get_file(file_path)
    pdf_bytes = blob_client.download_blob().content_as_bytes()
    #with open(file_path, "rb") as f:
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" ' \
                  f'width="800" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)   
    print("------"+st.session_state["file_name"]+" extract/load main table begin------")

    output_main_table_path = file_path[:-3] + 'csv'
    print('output_main_table_path:',output_main_table_path)
    if st.session_state['blob_client'].check_file_exist(output_main_table_path):
        st.text("主表格已被提取")
        st.text("读取主表格开始：") 
        print(f"文件 {output_main_table_path} 存在。")
        blob_client = st.session_state['blob_client'].get_file(output_main_table_path)
        csv_this = blob_client.download_blob()
        main_table = pd.read_csv(csv_this)
        #main_table = load_csv_data(csv_this)
        file_type  = get_file_type(main_table)
        df = pd.DataFrame(main_table.iloc[:,1:])
        st.session_state["file_type"] = file_type
        print("file_type:",st.session_state["file_type"])
        st.empty()
        st.text("读取主表格结束：") 
    else:
        st.text("主表格未被提取")
        st.text("提取主表格程序开始：") 
        print(f"文件 {output_main_table_path} 不存在。")
        main_table,file_type = extract_main_file_table(file_path)
        print(main_table)        
        df = pd.DataFrame(main_table[1:,], columns=['Receipt Date','No.','Expense Type','Cost Centre WBS', 'Have Rcpt','Tax Cde','Receipt Amount','Crcy'])  
        df.to_csv(output_main_table_path, encoding='utf_8_sig')
        with open(output_main_table_path, 'rb') as data:
            st.session_state['blob_client'].upload_file(data,output_main_table_path)        
        st.session_state["file_type"] = file_type
        print("file_type:",st.session_state["file_type"])
        st.empty()
        st.text("提取主表格程序结束") 

    st.empty()
    st.subheader("提取到的主表格信息如下：")
    #st.table(main_table)
    st.dataframe(df)
    #if 'chart_data' not in st.session_state:  # chart_data 存放主表格的数组 
    st.session_state.chart_data = df
    st.session_state.attach_num = df.shape[0]
    print("------"+st.session_state["file_name"]+" main table extract/load end------")
    
if __name__ == '__main__':
    main() 
