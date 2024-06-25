import os
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, generate_container_sas, ContentSettings
#from dotenv import load_dotenv

class AzureBlobStorageClient:
    def __init__(self, account_name: str = None, account_key: str = None, container_name: str = None):

        #load_dotenv()

        self.account_name : str = account_name if account_name else os.getenv('BLOB_ACCOUNT_NAME')
        self.account_key : str = account_key if account_key else os.getenv('BLOB_ACCOUNT_KEY')
        self.connect_str : str = f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"
        self.container_name : str = container_name if container_name else os.getenv('BLOB_CONTAINER_NAME')
        self.blob_service_client : BlobServiceClient = BlobServiceClient.from_connection_string(self.connect_str)

    def delete_file(self, file_name):
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        blob_client.delete_blob()
        
    def get_file(self, file_name):
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        return blob_client

    def upload_file(self, bytes_data, file_name, content_type='application/pdf'):
        # Create a blob client using the local file name as the name for the blob
        #print(file_name)
        #print(bytes_data)
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        # Upload the created file
        blob_client.upload_blob(bytes_data, overwrite=True, content_settings=ContentSettings(content_type=content_type))
        # Generate a SAS URL to the blob and return it
        return blob_client.url + '?' + generate_blob_sas(self.account_name, self.container_name, file_name,account_key=self.account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))
        
    def upload_image(self, bytes_data, file_name):
        # Create a blob client using the local file name as the name for the blob
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
        # Upload the created file
        blob_client.upload_blob(bytes_data, overwrite=True)
        # Generate a SAS URL to the blob and return it

    def get_all_files(self):
        # Get all files in the container from Azure Blob Storage
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blob_list = container_client.list_blobs(include='metadata')
        # sas = generate_blob_sas(account_name, container_name, blob.name,account_key=account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))
        sas = generate_container_sas(self.account_name, self.container_name,account_key=self.account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))
        files = []
        converted_files = {}
        for blob in blob_list:
            if not blob.name.startswith('converted/'):
                if blob.name.split('.')[1] == 'png' :  #do not include png files
                    continue
                files.append({
                    "filename" : blob.name,
                    "converted": blob.metadata.get('converted', 'false') == 'true' if blob.metadata else False,
                    "embeddings_added": blob.metadata.get('embeddings_added', 'false') == 'true' if blob.metadata else False,
                    "fullpath": f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob.name}?{sas}",
                    "converted_filename": blob.metadata.get('converted_filename', '') if blob.metadata else '',
                    "converted_path": ""
                    })
            else:
                converted_files[blob.name] = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob.name}?{sas}"

        for file in files:
            converted_filename = file.pop('converted_filename', '')
            if converted_filename in converted_files:
                file['converted'] = True
                file['converted_path'] = converted_files[converted_filename]
        
        return files
    
    def check_folder_exist(self,folder_name): #这里是文件夹的名字
        # Get all files in the container from Azure Blob Storage
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blob_list = container_client.list_blobs(include='metadata')
        #sas = generate_container_sas(self.account_name, self.container_name,account_key=self.account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))
        folder_name = folder_name + '/'
        for blob in blob_list:
            if blob.name.startswith(folder_name):
                return True
        
        return False
    
    def check_file_exist(self,file_name): #这里是包含文件类型的文件的名字
        # Get all files in the container from Azure Blob Storage
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blob_list = container_client.list_blobs(include='metadata')
        print('check file exist: ', file_name)
        #sas = generate_container_sas(self.account_name, self.container_name,account_key=self.account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))
        for blob in blob_list:
            if blob.name.startswith(file_name):
                return True
        
        return False
        
    def get_attach_file_number(self,folder_name): #这里是文件夹的名字
        # Get all files in the container from Azure Blob Storage
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blob_list = container_client.list_blobs(include='metadata')
        #sas = generate_container_sas(self.account_name, self.container_name,account_key=self.account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=3))
        str_name = folder_name + '/' + folder_name
        num = 0
        for blob in blob_list:
            if blob.name.startswith(str_name):
                num = num + 1
    
        return num
        
    def get_folder_files(self,folder_name): #这里是文件夹的名字
        # Get all files in the container from Azure Blob Storage
        container_client = self.blob_service_client.get_container_client(self.container_name)
        str_name = folder_name + '/'
        blob_list = container_client.list_blobs(name_starts_with=str_name)
        return blob_list

    def upsert_blob_metadata(self, file_name, metadata):
        blob_client = BlobServiceClient.from_connection_string(self.connect_str).get_blob_client(container=self.container_name, blob=file_name)
        # Read metadata from the blob
        blob_metadata = blob_client.get_blob_properties().metadata
        # Update metadata
        blob_metadata.update(metadata)
        # Add metadata to the blob
        blob_client.set_blob_metadata(metadata= blob_metadata)

    def get_container_sas(self):
        # Generate a SAS URL to the container and return it
        return "?" + generate_container_sas(account_name= self.account_name, container_name= self.container_name,account_key=self.account_key,  permission="r", expiry=datetime.utcnow() + timedelta(hours=1))

    def get_blob_sas(self, file_name):
        # Generate a SAS URL to the blob and return it
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{file_name}" + "?" + generate_blob_sas(account_name= self.account_name, container_name=self.container_name, blob_name= file_name, account_key= self.account_key, permission='r', expiry=datetime.utcnow() + timedelta(hours=10))
