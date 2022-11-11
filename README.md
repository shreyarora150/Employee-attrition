
# Employee-attrition

You are working as a data scientist with HR Department of a large insurance company focused on sales team attrition. Insurance sales teams help insurance companies generate new business by contacting potential customers and selling one or more types of insurance. The department generally sees high attrition and thus staffing becomes a crucial aspect.

To aid staffing, you are provided with the monthly information for a segment of employees for 2016 and 2017 and tasked to predict whether a current employee will be leaving the organization in the upcoming two quarters (01 Jan 2018 - 01 July 2018) or not, given:

Demographics of the employee (city, age, gender etc.)
Tenure information (joining date, Last Date)
Historical data regarding the performance of the employee (Quarterly rating, Monthly business acquired, designation, salary)

# Challenge 

Formulating the problem into a Machine Learning and transforming the data was the most challenging part. A snapshot of the data is provided under

![image](https://user-images.githubusercontent.com/42665120/201439109-24968014-816a-4d7e-804d-415956d85d53.png)

As the objective was to predict if an employee will leave the organization in the upcoming two quarters, the target variable was taken such that if an employee leaves the organization within 180 days of review it was taken was 1 and 0 otherwise i.e., if the last working day is 25-11-2017 and a review was conducted on 01-05-2017(208 days prior), target would be 0 and for the next review conducted on 01-06-2017(177 days prior), the target would be 1. The training data was taken only till 01-08-2017 as a full 180 days was required for prediction. The chalenge was to assess the level of data or each employe, wheher predictions had to be done at review level for each employee otherwise there would not be sufficient data and the changes in employee performace/behaviour might be difficult to catch if data was minimized to one row per employee.

# Feature Engineering

Following extra features were created along with the one's provided in the data set

1.Running average of business added 
2.Running average of quatery rating
3.Months since joining (dropped
4.Promotion rate (number of promotions/average time between promotions) 
5.Salary increment rate (salary increment/average time between promotions)

Encoded categorical data like gender,city and education level

# Model selection

1-0 loss function for evaulation

Trained and compared Logistic regression, Decision Tree, Random Forest, Gradient boosting and XGB classifier .Attained 72% score on private data set with XGB Classifier


