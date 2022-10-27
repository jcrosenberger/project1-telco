import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



'''
████████╗███████╗██╗░░░░░░█████╗░░█████╗░ 
╚══██╔══╝██╔════╝██║░░░░░██╔══██╗██╔══██╗ 
░░░██║░░░█████╗░░██║░░░░░██║░░╚═╝██║░░██║ 
░░░██║░░░██╔══╝░░██║░░░░░██║░░██╗██║░░██║ 
░░░██║░░░███████╗███████╗╚█████╔╝╚█████╔╝ 
░░░╚═╝░░░╚══════╝╚══════╝░╚════╝░░╚════╝░ 
'''

def helper():
    pass_into_prep_telco_data = 'train, validate, model = prep_telco_data(df)'
    pass_into_model_telco_data = 'x_train, y_train, x_validate, y_validate, x_test, y_test = model_telco_data(df)'
    list_of_main_functions = [pass_into_prep_telco_data, pass_into_model_telco_data]
    
    print('Main functions:',*list_of_main_functions, sep = '\n\n')


####################################################################################
####################           Main Function option 1           ####################
####################################################################################
##########  Function to prepare the Telco Data Frame and return clean df  ##########
####################################################################################


def prep_telco_data(df):

    '''
    Strips Data Frame down, adding dummy variables for several categories
    of services. It additionally removes unneeded variables. 
    '''
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)
    
    # Convert binary categorical variables to numeric
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # uses split_telco_data function to develop and return train, validate, and test variables
    train, validate, test = split_telco_data(df)
    

    return x_train, y_train, x_validate, y_validate, x_test, y_test 
    return train, validate, test


#################################################################################
####################         Main Function option 2          ####################
#################################################################################
##########    Function to prepare the Telco Data Frame for Modeling    ##########
#################################################################################


def model_telco_data(df):

    '''
    Strips Data Frame down, adding dummy variables for several categories
    of services. It additionally removes unneeded variables. 
    '''
    # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)

    # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']

    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)

    # Convert binary categorical variables to numeric
    df['is_female'] = df.gender.map({'Female': 1, 'Male': 0})
    df['has_partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['has_dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['has_phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['has_paperless_billing'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['did_churn'] = df.churn.map({'Yes': 1, 'No': 0})

    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                            'online_security', \
                            'online_backup', \
                            'device_protection', \
                            'tech_support', \
                            'streaming_tv', \
                            'streaming_movies', \
                            'contract_type', \
                            'internet_service_type', \
                            'payment_type']], dummy_na=False, \
                            drop_first=True)
    
    # Joins original dataframe with newly constructed dataframe using converted dummy variables
    df = pd.concat([df, dummy_df], axis = 1)

    # Creates dummy variables for customers based on which quartile their monthly charges fit
    df['charges_lower_quartile'] = df.monthly_charges <= df.monthly_charges.quantile(.25)
    df['charges_higher_quartile'] = df.monthly_charges >= df.monthly_charges.quantile(.75)
    dummy_df['mid_charge1'] = df.monthly_charges < df.monthly_charges.quantile(.75)
    dummy_df['mid_charge2'] = df.monthly_charges > df.monthly_charges.quantile(.25)
    df['mid_charge'] = dummy_df['mid_charge1'] == dummy_df['mid_charge2']





    #Drops variables which have had binary variables created to represent for machine learning 
    df = df.drop(columns=['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing',
                        'multiple_lines', 'online_security', 'online_backup', 'device_protection', 
                        'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 
                        'internet_service_type', 'payment_type', 'churn'])

    #Drops variables unused for models
    df = df.drop(columns=['monthly_charges', 'total_charges'])
    
    # uses split_telco_data function to develop and return train, validate, and test variables
    train, validate, test = split_telco_data(df)
    
    # 
    x_train, y_train, x_validate, y_validate, x_test, y_test = model(train, validate, test)

    return x_train, y_train, x_validate, y_validate, x_test, y_test 
    

################################################################################################################
########## Function to Take in Data Frame and then Split it into train, validate and test Data Frames ##########
################################################################################################################


def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state = 7, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state = 7, 
                                   stratify=train_validate.churn)
    return train, validate, test


####################################################################################################
########## Further Splits Data Frames into X and Y Groups for Machine Learning Algorithms ##########
####################################################################################################


def model(train, validate, test):
    ''' splitting data into two groups further. One group is going to be used to guess 
    the y outcome. The other group will know the y outcome and will be used to verify
    the model we create '''

    x_train = train.drop(columns=['churn'])
    y_train = train.churn

    x_validate = validate.drop(columns=['churn'])
    y_validate = validate.churn

    x_test = test.drop(columns=['churn'])
    y_test = test.churn

    return x_train, y_train, x_validate, y_validate, x_test, y_test 