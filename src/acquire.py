'''
██╗███╗░░░███╗██████╗░░█████╗░██████╗░████████╗ ███╗░░░███╗░█████╗░██████╗░██╗░░░██╗██╗░░░░░███████╗░██████╗
██║████╗░████║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝ ████╗░████║██╔══██╗██╔══██╗██║░░░██║██║░░░░░██╔════╝██╔════╝
██║██╔████╔██║██████╔╝██║░░██║██████╔╝░░░██║░░░ ██╔████╔██║██║░░██║██║░░██║██║░░░██║██║░░░░░█████╗░░╚█████╗░
██║██║╚██╔╝██║██╔═══╝░██║░░██║██╔══██╗░░░██║░░░ ██║╚██╔╝██║██║░░██║██║░░██║██║░░░██║██║░░░░░██╔══╝░░░╚═══██╗
██║██║░╚═╝░██║██║░░░░░╚█████╔╝██║░░██║░░░██║░░░ ██║░╚═╝░██║╚█████╔╝██████╔╝╚██████╔╝███████╗███████╗██████╔╝
╚═╝╚═╝░░░░░╚═╝╚═╝░░░░░░╚════╝░╚═╝░░╚═╝░░░╚═╝░░░ ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░░╚═════╝░╚══════╝╚══════╝╚═════╝░
'''

datasets_available= ['telco']
print('The following datasets are available:',*datasets_available, sep='\n')

#generally required for working with datasets
import seaborn as sns
import pandas as pd
import numpy as np
import os

#required for acquire
from src.env import get_db_url as get_db_url

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


##########################       Function to Import Data from CodeUp SQL Server       ##########################

########################## Requires env.py file which will be necessary to access SQL Server ##########################
'''
████████╗███████╗██╗░░░░░░█████╗░░█████╗░ 
╚══██╔══╝██╔════╝██║░░░░░██╔══██╗██╔══██╗ 
░░░██║░░░█████╗░░██║░░░░░██║░░╚═╝██║░░██║ 
░░░██║░░░██╔══╝░░██║░░░░░██║░░██╗██║░░██║ 
░░░██║░░░███████╗███████╗╚█████╔╝╚█████╔╝ 
░░░╚═╝░░░╚══════╝╚══════╝░╚════╝░░╚════╝░ 
'''

def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a DataFrame.
    It exists to run in the case that telco csv does not exist in proper folder
    as checked by get_telco_data function
    '''
    sql_query = '''
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                '''
                
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df

def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''
    if os.path.isfile('data/telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('data/telco_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('data/telco_df.csv')
        
    return df




