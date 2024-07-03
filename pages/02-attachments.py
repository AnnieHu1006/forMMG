import streamlit as st
import tempfile
import os
import time
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import webbrowser
#from attach_codes import extract_attach_keywords
from attach_codes import extract_one_attach_keywords,find_same_items,one_value_find_same_items

sslist = ["attach_table","tmp_t"]
for ss in sslist:
    if ss not in st.session_state:
        st.session_state[ss] = ""

if 'saved_attach' not in st.session_state:
    st.session_state.saved_attach = pd.DataFrame(index=range(30), columns=['file_name','Date', 'Type','Amount','Crcy','Accept_Invoice','Note'])
    for row in range(st.session_state.saved_attach.shape[0]):
        st.session_state.saved_attach.iloc[row,:] = None

if 'subattach_i' not in st.session_state:    
    st.session_state.subattach_i = 1   

# 读取CSV文件
@st.cache_data
def load_csv_data(file_path):
    data = pd.read_csv(file_path,sep=',')
    return data

def get_attach_filename(selected_df):
    filenames = []
    for index, row in selected_df.iterrows():
        if len(filenames) == 0:
            filenames.append(row['file_name'])
        elif row['file_name'] != filenames[-1]:
            filenames.append(row['file_name'])
    return filenames
    
def get_attach_file_number(folder_path):
    # 列出文件夹下所有的文件和文件夹
    entries = os.listdir(folder_path)     
    # 计算文件的数量
    number_of_files = 0
    for entry in entries:
        if os.path.isfile(os.path.join(folder_path, entry)):
            number_of_files += 1
    return number_of_files
    
@st.cache_data
def save_attach(saved_attach,subattach_i,file_name, subattach_sum):
    saved_attach.iloc[subattach_i-1,0] = file_name
    saved_attach.iloc[subattach_i-1,1:] = subattach_sum
    
def amount_to_float(amount_input):
    if np.isreal(amount_input):
        return float(amount_input)
        
    if amount_input!=None and amount_input!=' ':
        value = amount_input.replace(',', '')
        value = value.replace(" ", "") #去除掉字符串里的空格
        return float(value)  
    else:
        return 0.0

def highlight_first_row(s):
    return ['background-color: green' if i == 0 else '' for i in range(len(s))]

def main():
    print("------extract attachment function begin------")
    # 定义页面列表
    pages = {
        "attachments": "提取的附件的总内容",
        "details": "每一份附件的内容",
        "compare": "附件提取结果与主页结果做比较"
    }    
    page_names = list(pages.keys())
    # 在侧边栏选择页面
    selection = st.sidebar.radio("Go to", page_names)

    selected_columns = ['file_name','page_num','item_num','cls','score','Date', 'Type','Amount','Crcy','Accept_Invoice']
    attach_files_folder = st.session_state["file_name"] #output_attach_folder
    file_type = st.session_state["file_type"]
    output_attach_res_name = st.session_state["file_name"]+'-attach-extract-results.csv'
    output_attach_res_path = attach_files_folder + '/' + output_attach_res_name
    print(output_attach_res_path)
    if st.session_state['blob_client'].check_file_exist(output_attach_res_path):
        print(f"文件 {output_attach_res_name} 存在。")
        if st.session_state["selected_df"].empty:
            blob_client = st.session_state['blob_client'].get_file(output_attach_res_path)
            csv_this = blob_client.download_blob()
            df = pd.read_csv(csv_this)
            #df = load_csv_data(csv_this)
            selected_df = df[selected_columns]
            st.session_state["selected_df"] = selected_df 
        else:
            selected_df = st.session_state["selected_df"]           
        #这里如果是新读取的提取的内容，比较文件名，如果不一致，那所有 st.session_state.saved_attach 里的值设为None   20240609
        if st.session_state.saved_attach.iloc[0,0] != None:
            print('st.session_state["file_name"]:',st.session_state["file_name"])
            print('st.session_state.saved_attach.iloc[0,0]:',st.session_state.saved_attach.iloc[0,0])
            if st.session_state["file_name"] in st.session_state.saved_attach.iloc[0,0]:
                #print(st.session_state.saved_attach.iloc[0,0])
                #print(st.session_state["file_name"])
                print("st.session_state.saved_attach do not change to None")
            else:
                #print(st.session_state.saved_attach.iloc[0,0])
                #print(st.session_state["file_name"])
                print("st.session_state.saved_attach need to change to None")
                for row in range(st.session_state.saved_attach.shape[0]):
                    st.session_state.saved_attach.iloc[row,:] = None
                st.session_state.subattach_i = 1
    else:
        print(f"文件 {output_attach_res_name} 不存在，需要先提取附件信息！")
        print("attch folder path: ",attach_files_folder)
        print("file_type: ", file_type)
        st.text("提取所有附件关键词信息开始：")
        #设置进度条
        min_value = 0       # 设置进度条的最小值和最大值
        max_value = st.session_state['blob_client'].get_attach_file_number(attach_files_folder)
        process_word = st.empty()
        #progress_value = st.slider('Progress', min_value=min_value, max_value=float(max_value), value=min_value, key='progress')      # 设置进度条的当前值
        progress_bar = st.progress(min_value)       # 创建一个进度条
        process_i=1       # 进度条计数  
        mres = np.empty(shape=[0, 10], dtype=str) # 多加一个页数和一个页数上的item，也就是如果是pdf有多页,每页有多项的话，就会知道是第几页的第几项
        approve_flag = ""
        approve_path = ""
        temp_img_path = 'temp_attach_images'
        blobs = st.session_state['blob_client'].get_folder_files(attach_files_folder)
        for blob in blobs:
            #print(blob)
            process_word.text('正在提取第'+ str(process_i)+'个附件的关键词'+',共'+str(max_value)+'个附件')
            value = int(process_i*100/max_value+0.5)
            if value > 100:
                value = 100
            progress_bar.progress (value)     # 更新进度条的值
            mres,one_flag,one_path = extract_one_attach_keywords(blob.name,blob.size,file_type,mres,temp_img_path) #加入提取一个附件的关键词的函数，在attach_codes.py文件里
            if approve_flag == "":
                approve_flag = one_flag
                approve_path = one_path
            process_i = process_i+1      # 使用循环来更新进度条
        #progress_bar.empty()               # 进度条更新完毕后，可以将进度条从页面上移除
        #selected_df = extract_attach_keywords(attach_files_folder,file_type) 
        #将numpy数组转换为pandas DataFrame
        selected_df = pd.DataFrame(mres, columns=selected_columns)
        print(selected_df) 
        print(approve_flag,approve_path)
        #out_csv_path = os.path.join(attach_files_folder,'attach_results.csv')
        #selected_df.to_csv(out_csv_path, encoding='utf_8_sig',index=False)
        selected_df.to_csv(output_attach_res_name, encoding='utf_8_sig') 
        with open(output_attach_res_name, 'rb') as data:
            st.session_state['blob_client'].upload_file(data,output_attach_res_path)
        process_word.text('提取所有附件关键词信息结束')
        #因为是第一次提取，所以将所有 st.session_state.saved_attach 里的值设为None     20240609
        for row in range(st.session_state.saved_attach.shape[0]):
            st.session_state.saved_attach.iloc[row,:] = None
        st.session_state.subattach_i = 1
    
    filenamelist = get_attach_filename(selected_df)
    st.session_state.attach_num = len(filenamelist)
    #print(len(filenamelist))
    print(filenamelist[0])
    print(st.session_state["file_name"])
    
    # 在不点击保存附件的按钮的时候，对每个提取出来的结果，自动进行保存    20240630
    #for i in range(st.session_state.attach_num):
    #    st.session_state.saved_attach.iloc[i,0] = filenamelist[i]
    #    one_attach = selected_df[selected_df['file_name']==filenamelist[i]]
    #    print(one_attach)
    #    st.session_state.saved_attach.iloc[i,1] = one_attach['Date'].iloc[0] # Type
    #    st.session_state.saved_attach.iloc[i,2] = one_attach['Type'].iloc[0] # Type
    #    st.session_state.saved_attach.iloc[i,3] = float(one_attach['Amount'].sum()) # Amount
    #    st.session_state.saved_attach.iloc[i,4] = one_attach['Crcy'].iloc[0] # Crcy
    #    st.session_state.saved_attach.iloc[i,5] = one_attach['Accept_Invoice'].iloc[0] # Accept_Invoice
    #print(st.session_state.saved_attach)
    
    if selection == page_names[0]:
        st.subheader('提取的附件信息汇总')
        st.dataframe(selected_df)
    elif selection == page_names[1]:
        st.subheader("每个附件的详细信息核对")
        #展示附件的缩略图      
        print("filename list number : " + str(len(filenamelist)))
        thumbnail_id = 0
        col_num = 7
        row_num = int(len(filenamelist)/col_num) + 1
        print("col_num,row_num : ", col_num, row_num)
        for i in range(row_num):
            thiscolums = st.columns(col_num)
            for j,col in enumerate(thiscolums):
                with col:
                    if thumbnail_id >=len(filenamelist):
                        print("col break")
                        break
                    if len(st.session_state["thumbnail"])>thumbnail_id:
                        st.image(st.session_state["thumbnail"][thumbnail_id],caption = thumbnail_id+1)
                        thumbnail_id = thumbnail_id +1
                    else:
                        file_name = filenamelist[thumbnail_id]
                        if file_name.endswith('.pdf'):
                            file_path = 'temp_attach_images/' + file_name[:-4]+'_1.png'
                        elif file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            file_path = attach_files_folder + '/' + file_name     
                        print(file_path)
                        blob_client = st.session_state['blob_client'].get_file(file_path)
                        img_this = blob_client.download_blob()
                        thisimage = Image.open(img_this)
                        thumbnail_img = thisimage.resize((200, 200)) 
                        thumbnail_id = thumbnail_id +1
                        st.session_state["thumbnail"].append(thumbnail_img)
                        st.image(thumbnail_img,caption = thumbnail_id)
            if thumbnail_id >=len(filenamelist):
                print("ROW break")
                break
                
        # 获取第i个子附件 
        #thiscolums = st.columns(len(filenamelist))
        #for i, column in enumerate(thiscolums):
        #    with column:
        #        st.write(f"{i+1}")
        # 生成数据
        #x_values = np.linspace(-10, 10, 200)  # 横线的x坐标
        #y_values = np.full_like(x_values, 0)  # 纵坐标，这里是0，表示横线        
        #fig, ax = plt.subplots()
        #ax.plot(x_values, y_values, color='black')  # 绘制横线        
        #ax.axis('off')
        #st.pyplot(fig)
        # 分别用按钮的下一个的方式，和进度条里自选的方式，获取第i个子附件  20240609
        subattach_i = st.session_state.subattach_i
        if len(filenamelist) > 1:                 
            if st.button('下一个子附件') and st.session_state.subattach_i < len(filenamelist):
                next_i = st.session_state.subattach_i + 1
                print('next_i:',next_i)
                subattach_i = st.slider("选择子附件：", min_value=1, max_value=len(filenamelist), value=next_i)
                st.session_state.subattach_i = subattach_i
                print('subattach_i:',subattach_i)
                print('st.session_state.subattach_i:',st.session_state.subattach_i)
            elif st.session_state.subattach_i ==1 :
                subattach_i = st.slider("选择子附件：", min_value=1, max_value=len(filenamelist),value=1) 
            else:
                if st.session_state.subattach_i <= len(filenamelist):
                    print('st.session_state.subattach_i:',st.session_state.subattach_i)
                    st.session_state.subattach_i = subattach_i
                    subattach_i = st.slider("选择子附件：", min_value=1, max_value=len(filenamelist),value=st.session_state.subattach_i)                    
                    print('subattach_i:',subattach_i)
                else:
                    subattach_i = st.slider("选择子附件：", min_value=1, max_value=len(filenamelist),value=1) 
                    st.session_state.subattach_i = subattach_i 
         
        # 对第i个子附件的显示   
        if  subattach_i >   len(filenamelist):
            subattach_i = 1
            st.session_state.subattach_i = subattach_i
        st.session_state.subattach_i = subattach_i    
        with st.container():
            file_name = filenamelist[subattach_i-1]
            file_path = attach_files_folder + '/' + file_name
            print(file_path)
            # 每个子附件的标题
            st.subheader(f"子附件  {filenamelist[subattach_i-1]}")
            # 在这里添加子附件的内容
            st.write(f"这是第 {subattach_i} 个子附件的内容：")          
            attach_i_df = selected_df[selected_df['file_name']==file_name]
            attach_i_df = attach_i_df.iloc[:,1:]
            #st.dataframe(attach_i_df)
            chart_i_df = st.data_editor(attach_i_df,key = 'changed_rows',
                    column_config={
                        "Type": st.column_config.SelectboxColumn(
                            "Type",
                            help="The category of the attach item",
                            width="medium",
                            options=[
                                "Accommodation(DOM)",
                                "Accommodation(INT)",
                                "Airfares (DOM)",
                                "Airfares (INT)",
                                "Meals Travelling (Employee)",
                                "Meal NonTravel OffSite Employ",
                                "Meal NonTravel Offsite NonEmpl",
                                "Taxi/CarHire/Incidentals (DOM)",
                                "Taxi/CarHire/Incidentals (INT)",
                                "Office & Admin Supplies",
                                "Travel Health & Medicals",
                                "Licences and Permits",
                                "Training-courses, Training, Seminar, Conferences",
                                "Telephone Subsidies (Business)",
                                "Employee Memberships/Subscript"
                                
                            ],
                        ),
                        "Accept_Invoice": st.column_config.SelectboxColumn(
                            "Accept_Invoice",
                            help="The category of the attach item",
                            width="medium",
                            options=[
                                "Yes",
                                "No"
                            ],
                        )
                     },
                num_rows="dynamic", disabled=['file_name','page_num','item_num','cls', 'score'],hide_index=True)
            
            #print("chart_i_df ",chart_i_df.iloc[:,5])
            st.write(f"这是第 {subattach_i} 个子附件的合计结果：") 
            subattach_sum = chart_i_df.iloc[0:1,4:9]
            sum_amount = 0.0
            print("edited_rows:",st.session_state["changed_rows"]["edited_rows"])
            print("added_rows:",st.session_state["changed_rows"]["added_rows"])
            print("deleted_rows:",st.session_state["changed_rows"]["deleted_rows"])
            if len(st.session_state["changed_rows"]["edited_rows"]) == 0:
                for row in range(len(chart_i_df)):
                    value = amount_to_float(chart_i_df['Amount'].iloc[row])
                    sum_amount = sum_amount + value
            else:
                for row in range(len(chart_i_df)):
                    modifed_value = st.session_state["changed_rows"]["edited_rows"].get(row,'None')
                    if modifed_value!= 'None' and 'Amount' in modifed_value:
                        sum_amount = sum_amount + float(modifed_value['Amount'])
                    else:
                        value = amount_to_float(chart_i_df['Amount'].iloc[row])
                        sum_amount = sum_amount + value
            if 'Amount' in st.session_state["changed_rows"]["added_rows"]:  # 防止刷新时的bug,   20240702
                #print('st.session_state["changed_rows"]["added_rows"]',len(st.session_state["changed_rows"]["added_rows"]))
                for row in range(len(st.session_state["changed_rows"]["added_rows"])):
                    value = st.session_state["changed_rows"]["added_rows"][row]['Amount']
                    sum_amount = sum_amount + value
                
            #sum_amount = chart_i_df.iloc[:,6].sum()
            subattach_sum.iloc[0,2] = sum_amount
            subattach_sum.iloc[0,4] = chart_i_df.iloc[0,8]
            st.dataframe(subattach_sum,hide_index=True)
            
            if st.button('确定保存'):
                print('save keywords of file: ',file_name)
                st.session_state.saved_attach.iloc[subattach_i-1,0] = file_name
                for idx in range(5):
                    st.session_state.saved_attach.iloc[subattach_i-1,idx+1] = subattach_sum.iloc[0, idx]
                #print(st.session_state.saved_attach.iloc[subattach_i-1,5])
                #print(subattach_sum.iloc[0, 4])
                subattach_i = st.session_state.subattach_i 
                print(st.session_state.saved_attach)
 
            if st.button('查验定额发票真伪'): #待实现！！！！！！
                #webbrowser.open('https://inv-veri.chinatax.gov.cn/index.html?source=bdjj&e_keywordid2=92056973642')
                driver.get("https://inv-veri.chinatax.gov.cn/index.html?source=bdjj&e_keywordid2=92056973642")
            
            #展示附件内容
            if file_name[-3:] == 'pdf':
                print("Attachment: show pdf attach")                
                blob_client = st.session_state['blob_client'].get_file(file_path)
                pdf_bytes = blob_client.download_blob().content_as_bytes()
                #with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
                              f'width="800" height="1000" type="application/pdf">'
                st.markdown(pdf_display, unsafe_allow_html=True) 
            elif file_name[-3:] == 'png' or file_name[-3:] == 'jpg':
                print("Attachment: show image attach")
                #st.image(file_path)
                blob_client = st.session_state['blob_client'].get_file(file_path)
                img_this = blob_client.download_blob()
                img = Image.open(img_this)
                st.image(img, caption='Attachment image', use_column_width=True)
    elif selection == page_names[2]:                    
            st.write(f"这是提取的附件的结果：") 
            df_extract_edit = st.session_state.saved_attach.drop('Date', axis=1) # 隐藏Data列    20240609
            df_init_edit = st.session_state.chart_data.drop(['Cost Centre WBS'], axis=1) # 隐藏Data列    20240609
            ext_list,init_list = find_same_items(df_extract_edit,st.session_state.attach_num,df_init_edit) #对提取的结果，和主表进行比对   20240615
            print('init_list',init_list)
            print('st.session_state["init_list"]',st.session_state["init_list"])
            print('st.session_state["ext_list"]',st.session_state["ext_list"])
            if st.session_state["ext_list"]==[]:  
                st.session_state["ext_list"] = ext_list
                st.session_state["init_list"] = init_list
            
            if 'Co' not in df_extract_edit.columns: #加入一列，是比对成功的，主表上的对应项    20240615
                df_extract_edit.insert(loc=0, column ='Co', value=[None] * len(df_extract_edit)) 
            if 'Select' not in df_extract_edit.columns: #加入一列，是用于选择行的，主表上的对应项    20240630
                df_extract_edit.insert(loc=0, column ='Select', value=[False] * len(df_extract_edit))                

            for i in range(len(st.session_state["ext_list"])):
                df_extract_edit['Co'].iloc[st.session_state["ext_list"][i]] = st.session_state["init_list"][i] + 1 
            confirmed_extract_df = st.data_editor(df_extract_edit.iloc[0:st.session_state.attach_num, :],hide_index=True)              

            # 定义一个按钮来触发求和操作，以及和主表的行的比较
            if st.button('求和并比对'):
                selected_rows = confirmed_extract_df[confirmed_extract_df['Select'] == True]
                rindices = np.where(confirmed_extract_df['Select'] == True) # Select的行的索引   
                st.session_state["select_sum"]=sum_values = selected_rows['Amount'].sum()
                st.write(f'Sum of selected values: {st.session_state["select_sum"]}') 
                init_row = one_value_find_same_items(selected_rows,sum_values,df_init_edit,st.session_state["init_list"]) #对求和的结果，和主表的剩余项进行比对   20240615
                if init_row !=-1:
                    for i in range(len(rindices[0])):
                        print('row_index',rindices[0][i])
                        st.session_state["init_list"].append(init_row)
                        st.session_state["ext_list"].append(rindices[0][i]) 
                        if 'Meal' in df_extract_edit['Type'].iloc[rindices[0][i]]: # 改Meal 类型                           
                            st.session_state.saved_attach['Type'].iloc[rindices[0][i]] = df_init_edit['Expense Type'].iloc[init_row]
                st_autorefresh(interval=1, limit=10, key="on_sum_button")
 
            if st.button('确定保存'): 
                output_confirmed_table_name = st.session_state["file_name"]+'-attach-confirmed-results.csv'
                output_confirmed_table_path = attach_files_folder + '/' + output_confirmed_table_name
                confirmed_extract_df.to_csv(output_confirmed_table_name, encoding='utf_8_sig',index=False)  
                with open(output_confirmed_table_name, 'rb') as data:
                    st.session_state['blob_client'].upload_file(data,output_confirmed_table_path)                

            st.write(f"这是主页总表的结果：")
            #st.dataframe(df_init_edit,hide_index=True)   
            html_str = df_init_edit.to_html(index=False) #对主表加入，比对成功的行修改颜色的显示   20240616
            html_rows = html_str.split('</tr>') # 按照行进行拆分
            #print(html_rows[0])
            for i in range(len(st.session_state["ext_list"])):   
                html_rows[st.session_state["init_list"][i]+1] = html_rows[st.session_state["init_list"][i]+1].replace('<tr>', '<tr style="background-color:lightblue;">') #background-color:lightblue  #color: blue
                #print(df_extract_edit['Co'].iloc[ext_list[i]])         
            html_str = '</tr>'.join(html_rows) # 将修改后的行组合到一起            
            st.markdown(html_str, unsafe_allow_html=True)
            

if __name__ == '__main__':
    main() 