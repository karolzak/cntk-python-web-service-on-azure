from __future__ import print_function
import os
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
   
def download_file(filename, file_url):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(file_dir, filename)
    if not os.path.exists(file_path):
        print('Downloading file from ' + file_url + ', may take a while...')
        urlretrieve(file_url,file_path)
        print('Saved file as ' + file_path)
    else:
        print('File already available at ' + file_path)

if __name__ == '__main__':
    download_file('HotailorPOC2.model','https://privdatastorage.blob.core.windows.net/github/cntk-python-web-service-on-azure/HotailorPOC2.model')
    download_file('HotailorPOC2_class_map.txt','https://privdatastorage.blob.core.windows.net/github/cntk-python-web-service-on-azure/HotailorPOC2_class_map.txt')