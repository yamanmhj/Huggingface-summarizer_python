from NLP_MODULE.logger import logging
from NLP_MODULE.exception import CustomeException
import sys
import os
logging.info("Starting script")
from NLP_MODULE.configuration.gcloud_data_sync import GCloudConnectionCURD



def sync_folder_from_gcloud(gcp_bucket_url, filename, destination):

        command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"        
        # command = f"gcloud storage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)
        
        
sync_folder_from_gcloud("nlpdataset","dataset.zip","dataset/dataset.zip")
