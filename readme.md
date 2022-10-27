## Telco Analysis Anticipating Why Customers Churned

## Description of Project

'Telco' is a well known, well formated dataset that many burgeoning datascientists use for training or practice purposes. The dataset is used because it came from a real telecommunications firm who was interested in figuring out which of its customers churned and when it was that they would do it, given conditions of the firm's services. 


### Modelling methods to test:
- knn
- random forest
- decision trees


### Hypothesis from theory:
- tenure will have negative correlation with churning (longer a customer, less likely to leave)
- contract types will have different effects on churning. I'm not sure which will positively correlate or negatively correlate to churning
- price will have a positive correlation with churning (the higher the price, the more likely a person is to leave)
- I suspect though, that a longer tenure will have a moderating effect on price. That is to say, if we control for tenure, price will show to have more extreme effects on customers who have shorter tenures and it will have a smaller impact on customers who have longer tenures with the telco firm. 




## How to reproduce:
- Must have credentials which grant access to the www facing codeup sql server.
- By design, I put a library of modules in the src folder.
- The acquire module has a dependency in the env module which must operate correctly. 
>> A module named env.py is required to function properly. That module must contain the following function:  
    
    def get_db_url(df):

    url = f'mysql+pymysql://{user}:{password}@{host}/{df}'
    return url 
    
- Additionally in the env.py module, user, password, and host must be defined for the function to work properly