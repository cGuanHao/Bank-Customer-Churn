#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

''' importing all the required ML packages：   
    # Logistic Regression
    # Support Vector Machines
    # Random Forrest
    # KNN or k-Nearest Neighbors  
    # Naive Bayes classifier
    # Decision Tree    
    # Perceptron
    # Stochastic Gradient Descent 
    
    # train_test_split
    # metrics '''
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


#%% Exploratory Data Analysis(EDA)

start_time = time.time()

''' Load the csv '''
data = pd.read_csv(r'..\bank-customer-churn/Customer-Churn-Records.csv')
print(data.head())
print(data.tail())

''' Check the Dtype and if exist null '''
print(data.info())
''' Observation：
    Thankfully there's no null, so we don't need to deal with it. '''

''' Check the narrative statistics and the data column '''
print(data.describe())
print(data.columns)

''' The detail about the dataset columns from kaggle:
    RowNumber—corresponds to the record (row) number and has no effect on the output.
    CustomerId—contains random values and has no effect on customer leaving the bank.
    Surname—the surname of a customer has no impact on their decision to leave the bank.
    
    CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
    Geography—a customer’s location can affect their decision to leave the bank.
    Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
    Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
    Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
    Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
    NumOfProducts—refers to the number of products that a customer has purchased through the bank.
    HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
    IsActiveMember—active customers are less likely to leave the bank.
    EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
    Complain—customer has complaint or not.
    Satisfaction Score—Score provided by the customer for their complaint resolution.
    Card Type—type of card hold by the customer.
    Points Earned—the points earned by the customer for using credit card.
    
    Exited—whether or not the customer left the bank. '''    
''' Observation：
    We can expect that "RowNumber", "CustomerId" and "Surname", 
    the three values aren't relevant to our goal, so we will remove them later.
    
    Our goal is to except whether or not the client will leave the bank. 
    So an observation of correlation between "Exited" and remaining values will be critical. '''


# Analysing The Categorical Feature

''' Geography '''
print(data.groupby(['Geography', 'Exited'])['Exited'].count())
plt.gca().set_title('Variable Geography')
sns.countplot(x='Geography', hue='Exited', data=data)

''' Observation：
    French tended to not leave the bank, and German was more likely leaving the bank. '''

''' Gender '''
print(data.groupby(['Gender', 'Exited'])['Exited'].count())
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data[['Gender', 'Exited']].groupby(['Gender']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Exited vs Gender')
sns.countplot('Gender', hue='Exited', data=data, ax=ax[1])
ax[1].set_title('Gender: Exited vs Staied')
plt.show()
''' Observation：
    Women was more likely leaving the bank than men. '''

''' Tenure '''
print(data.groupby(['Tenure', 'Exited'])['Exited'].count())
f, ax = plt.subplots(1, 2, figsize=(20, 8))
data[['Tenure', 'Exited']].groupby(['Tenure']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Exited vs Tenure')
sns.countplot('Tenure', hue='Exited', data=data, ax=ax[1])
ax[1].set_title('Tenure: Exited vs Staied')
plt.show()
'''Observation：
   It was surprising to find there was at least 15% people left the bank in every tenure. 
   Exited rate decreased over tenure, but overall it was not quite significant.'''

''' NumOfProducts'''
print(data.groupby(['NumOfProducts', 'Exited'])['Exited'].count())
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data[['NumOfProducts', 'Exited']].groupby(['NumOfProducts']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Exited vs NumOfProducts')
sns.countplot('NumOfProducts', hue='Exited', data=data, ax=ax[1])
ax[1].set_title('NumOfProducts: Exited vs Staied')
plt.show()
'''Observation：
   It was shocked that the truth was totally opposite with our intuition. 
   The more products client have, the more they likely left the bank. '''

''' Satisfaction Score'''
print(data.groupby(['Satisfaction Score', 'Exited'])['Exited'].count())
f, ax = plt.subplots(1, 2, figsize=(20, 8))
data[['Satisfaction Score', 'Exited']].groupby(['Satisfaction Score']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Exited vs Satisfaction Score')
sns.countplot('Satisfaction Score', hue='Exited', data=data, ax=ax[1])
ax[0].set_title('Satisfaction Score: Exited vs Staied')
plt.show()
''' Observation：
    It was different from our expectation, the higher score could not prevent people left the bank. '''


# Analysing The Continous Feature

''' CreditScore '''
print('Highest CreditScore was:', data['CreditScore'].max())
print('Lowest CreditScore was:', data['CreditScore'].min())
print('Average CreditScore was:', data['CreditScore'].mean())
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.histplot(x=data['CreditScore'], kde=False, ax=ax[0])
ax[0].set_title("Histplot CreditScore", fontdict={'fontsize': 20})
sns.boxplot(x=data["CreditScore"], ax=ax[1])
ax[1].set_title("Boxplot CreditScore", fontdict={'fontsize': 20})
plt.show()
''' Observation：
    CreditScore has practically the non-perfect normal distribution.  
    Looking at the boxplots of our continuous variables, 
    we can see that we have few outliers, for now we will leave them. '''
    
''' Age '''
print('Highest Age was:', data['Age'].max(), 'Years')
print('Lowest Age was:', data['Age'].min(), 'Years')
print('Average Age was:', data['Age'].mean(), 'Years')
f, ax = plt.subplots(1, 3, figsize=(25, 8))
sns.histplot(x=data['Age'], kde=False, ax=ax[0])
ax[0].set_title('Histplot Age', fontdict={'fontsize': 20})
sns.boxplot(x=data["Age"], ax=ax[1])
ax[1].set_title('Boxplot Age', fontdict={'fontsize': 20})
sns.violinplot('Gender', 'Age', hue='Exited', data=data, split=True, ax=ax[2])
plt.show()
''' Observation：
    Male and female trend are the same in the every age. 
    The data show that every 10 ages increased would appear a leaving peak roughly since 30. 
    The highest peak was on the 40 ages. Apart from this, the distribution was normal distribution approximately. '''

''' Balance '''
print('Highest Balance was:', data['Balance'].max())
print('Lowest Balance was:', data['Balance'].min())
print('AverBalance Balance was:', data['Balance'].mean())
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.histplot(x=data['Balance'], kde=False, ax=ax[0])
ax[0].set_title('Histplot Balance', fontdict={'fontsize': 20})
sns.boxplot(x=data["Balance"], ax=ax[1])
ax[1].set_title('Boxplot Balance', fontdict={'fontsize': 20})
plt.show()
''' Observation：
    Most data was from 0 balance. The remaining data shows us a perfect normal distribution. '''

''' EstimatedSalary '''
print('Highest EstimatedSalary was:', data['EstimatedSalary'].max())
print('Lowest EstimatedSalary was:', data['EstimatedSalary'].min())
print('AverEstimatedSalary EstimatedSalary was:', data['EstimatedSalary'].mean())
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.histplot(x=data['EstimatedSalary'], kde=False, ax=ax[0])
ax[0].set_title('Histplot EstimatedSalary', fontdict={'fontsize': 20})
sns.boxplot(x=data["EstimatedSalary"], ax=ax[1])
ax[1].set_title('Boxplot EstimatedSalary', fontdict={'fontsize': 20})
plt.show()
''' Observation：
    EstimatedSalary looks like a uniform distribution. '''
    
''' Point Earned '''
print('Highest Point Earned was:', data['Point Earned'].max())
print('Lowest Point Earned was:', data['Point Earned'].min())
print('AverPoint Earned Point Earned was:', data['Point Earned'].mean())
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.histplot(x=data['Point Earned'], kde=False, ax=ax[0])
ax[0].set_title('Histplot Point Earned', fontdict={'fontsize': 20})
sns.boxplot(x=data["Point Earned"], ax=ax[1])
ax[1].set_title('Boxplot Point Earned', fontdict={'fontsize': 20})
plt.show()
''' Observation：
    It also looks like a uniform distribution. '''


# Analysing The Discrete Feature

''' HasCrCard '''
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.barplot('HasCrCard', 'Exited', data=data, ax=ax[0])
ax[0].set_title('HasCrCard vs Exited')
sns.countplot(x='HasCrCard', hue='Exited', data=data, ax=ax[1])
ax[1].set_title('Variable HasCrCard')
plt.show()
''' Observation：
    Whether having a credit card or not, it seems not affect the Exited Feature. '''

''' IsActiveMember '''
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.barplot('IsActiveMember', 'Exited', data=data, ax=ax[0])
ax[0].set_title('IsActiveMember vs Exited')
sns.countplot(x='IsActiveMember', hue='Exited', data=data, ax=ax[1])
ax[1].set_title('Variable IsActiveMember')
plt.show()
''' Observation：
    As our expectation, inactive member more likely to leave. '''

''' Complain '''
f, ax = plt.subplots(1, 2, figsize=(23, 8))
sns.barplot('Complain', 'Exited', data=data, ax=ax[0])
ax[0].set_title('Complain vs Exited')
sns.countplot(x='Complain', hue='Exited', data=data, ax=ax[1])
ax[1].set_title('Variable Complain')
plt.show()
''' Observation：
    Surprisingly, the Complain Feature and the Exited are almost 100% equal. 
    We have to remove the Complain Feature later, since the 100% equal are not healthy in ML. '''

''' Exited '''
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Exited'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Exited')
ax[0].set_ylabel('')
sns.countplot('Exited', data=data, ax=ax[1])
ax[1].set_title('Exited')
plt.show()
''' Observation：
    We observed that roughly 20% customers left the bank. 
    However, we still need to dig into more to find insight. '''


# Analysing The Ordinal Feature

''' Card Type '''
f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.countplot(x='Card Type', hue='Exited', data=data, ax=ax[0])
ax[0].set_title('Variable Card Type')
sns.barplot('Card Type', 'Exited', data= data, ax=ax[1])
ax[1].set_title('Card Type vs Exited')
plt.show()
''' Observation：
    Every level is much the same in Exited rate and number. '''


#%% Feature Engineering and Data Cleaning
 
''' Covert the object Dtype into numerical Dtype. '''
print(np.unique(data['Geography']))
print(np.unique(data['Gender']))
print(np.unique(data['Card Type']))
data['Geography'].replace(['France', 'Germany', 'Spain'], [0,1,2], inplace=True)
data['Gender'].replace(['Female', 'Male'], [0,1], inplace=True)
data['Card Type'].replace(['SILVER', 'GOLD', 'PLATINUM', 'DIAMOND'], [0,1,2,3], inplace=True)

''' Delete the useless features '''
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

''' Visualization
    Next step, we are going to establish a chart of those values in order to understand
    how they affect each other. It will be a benefit to following modeling. '''
sns.heatmap(data.corr(), annot=True, cmap='YlOrBr', fmt='.2f', linewidths=0.2)
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()
''' Observation：
    We can see that we have a 100% correlation between the Exited variable and the Complain variable as we expected before, 
    so let's eliminate the Complain variable. '''

''' Delete the highly overlapping feature. '''
data.drop(['Complain'], axis=1, inplace=True)


''' CreditScore_band '''
data['CreditScore_band'] = 0
data.loc[(data['CreditScore']>=300) & (data['CreditScore']<=579), 'CreditScore_band'] = 0
data.loc[(data['CreditScore']>=580) & (data['CreditScore']<=669), 'CreditScore_band'] = 1
data.loc[(data['CreditScore']>=670) & (data['CreditScore']<=739), 'CreditScore_band'] = 2
data.loc[(data['CreditScore']>=740) & (data['CreditScore']<=799), 'CreditScore_band'] = 3
data.loc[data['CreditScore']>=800, 'CreditScore_band'] = 4
print(data['CreditScore_band'])

''' Age_band '''
data['Age_band'] = pd.qcut(data['Age'], 5)
print(data[['Age', 'Age_band']].head(5))
print(np.unique(data['Age_band']))
   
data['Age_band'] = 0
data.loc[(data['Age']>17.999) & (data['Age']<=31), 'Age_band'] = 0
data.loc[(data['Age']>31) & (data['Age']<=35), 'Age_band'] = 1
data.loc[(data['Age']>35) & (data['Age']<=40), 'Age_band'] = 2
data.loc[(data['Age']>40) & (data['Age']<=46), 'Age_band'] = 3
data.loc[data['Age']>46, 'Age_band'] = 4
print(data['Age_band'])

''' Balance_band '''
data['Balance_band'] = pd.cut(data['Balance'], 4)
print(data[['Balance', 'Balance_band']].head(5))
print(np.unique(data['Balance_band']))

data['Balance_band'] = 0
data.loc[data['Balance']==0, 'Balance_band'] = 0
data.loc[(data['Balance']>0) & (data['Balance']<=62724.522), 'Balance_band'] = 1
data.loc[(data['Balance']>62724.522) & (data['Balance']<=125449.045), 'Balance_band'] = 2
data.loc[(data['Balance']>125449.045) & (data['Balance']<=188173.568), 'Balance_band'] = 3
data.loc[(data['Balance']>188173.568) & (data['Balance']<=250898.09), 'Balance_band'] = 4
print(data['Balance_band'])

''' EstimatedSalary_band '''
data['EstimatedSalary_band'] = pd.cut(data['EstimatedSalary'], 5)
print(data[['EstimatedSalary', 'EstimatedSalary_band']].head(5))
print(np.unique(data['EstimatedSalary_band']))

data['EstimatedSalary_band'] = 0
data.loc[(data['EstimatedSalary']<40007.76), 'EstimatedSalary_band'] = 0
data.loc[(data['EstimatedSalary']>40007.76) & (data['EstimatedSalary']<=80003.94), 'EstimatedSalary_band'] = 1
data.loc[(data['EstimatedSalary']>80003.94) & (data['EstimatedSalary']<=120000.12), 'EstimatedSalary_band'] = 2
data.loc[(data['EstimatedSalary']>120000.12) & (data['EstimatedSalary']<=159996.3), 'EstimatedSalary_band'] = 3
data.loc[(data['EstimatedSalary']>159996.3), 'EstimatedSalary_band'] = 4
print(data['EstimatedSalary_band'])

''' Point Earned_band '''
data['Point Earned_band'] = pd.qcut(data['Point Earned'], 5)
print(data[['Point Earned', 'Point Earned_band']].head(5))
print(np.unique(data['Point Earned_band']))
   
data['Point Earned_band'] = 0
data.loc[(data['Point Earned']<=370.0), 'Point Earned_band'] = 0
data.loc[(data['Point Earned']>370.0) & (data['Point Earned']<=529.0), 'Point Earned_band'] = 1
data.loc[(data['Point Earned']>529.0) & (data['Point Earned']<=682.0), 'Point Earned_band'] = 2
data.loc[(data['Point Earned']>682.0) & (data['Point Earned']<=840.0), 'Point Earned_band'] = 3
data.loc[data['Point Earned']>840.0, 'Point Earned_band'] = 4
print(data['Point Earned_band'])

''' Delete the replaced feature. '''
data.drop(['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Point Earned'], axis=1, inplace=True)


#%% Predictive Modeling

''' Separating features variables and the target variable. '''
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Exited'])
train_X = train.drop(['Exited'], axis=1, inplace=False)
train_Y = train['Exited']
test_X = test.drop(['Exited'], axis=1, inplace=False)
test_Y = test['Exited']
X = data.drop(['Exited'], axis=1, inplace=False)
Y = data['Exited']

# logistic regression
logreg = LogisticRegression()
logreg.fit(train_X, train_Y)
pred_log = logreg.predict(test_X)
acc_log = round(metrics.accuracy_score(pred_log, test_Y)*100, 2)
print('The accuracy of the Logistic Regression is', acc_log)

coeff_df = pd.DataFrame(X.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation', ascending=False))

# Radial Support Vector Machines(rbf-SVM)
rbf_SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)
rbf_SVM.fit(train_X, train_Y)
pred_rbf_SVM = rbf_SVM.predict(test_X)
acc_rbf = round(metrics.accuracy_score(pred_rbf_SVM, test_Y)*100, 2)
print('Accuracy for rbf SVM is ', acc_rbf)

# Linear Support Vector Machine(linear-SVM)
linear_SVM = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
linear_SVM.fit(train_X, train_Y)
pred_linear_SVM = linear_SVM.predict(test_X)
acc_linear = round(metrics.accuracy_score(pred_linear_SVM, test_Y)*100, 2)
print('Accuracy for linear SVM is', acc_linear)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_Y)
pred_random_forest = random_forest.predict(test_X)
acc_random_forest = round(metrics.accuracy_score(pred_random_forest, test_Y)*100, 2)
print('The accuracy of the Random Forests is', acc_random_forest)

# KNN
a_index = list(range(1, 11))
a = pd.Series()
x = [0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1, 11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = round(metrics.accuracy_score(prediction, test_Y)*100, 2)
    a = a.append(pd.Series(accuracy))

plt.plot(a_index, a)
plt.xticks(x)
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.show()
print('Accuracies for different values of n are:', a.values, 'with the max value as ', a.values.max())
''' The max accuracy value is 80.23 '''

# knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(train_X, train_Y)
pred_knn = knn.predict(test_X)
acc_knn = round(metrics.accuracy_score(pred_knn, test_Y)*100, 2)
print('The accuracy of the KNN is', acc_knn)

# Naive bayes
gaussian = GaussianNB()
gaussian.fit(train_X, train_Y)
pred_gaussian = gaussian.predict(test_X)
acc_gaussian = round(metrics.accuracy_score(pred_gaussian, test_Y)*100, 2)
print('The accuracy of the NaiveBayes is', acc_gaussian)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)
pred_decision_tree = decision_tree.predict(test_X)
acc_decision_tree = round(metrics.accuracy_score(pred_decision_tree, test_Y)*100, 2)
print('The accuracy of the Decision Tree is', acc_decision_tree)

# Perceptron
perceptron = Perceptron()
perceptron.fit(train_X, train_Y)
pred_perceptron = perceptron.predict(test_X)
acc_perceptron = round(metrics.accuracy_score(pred_perceptron, test_Y)*100, 2)
print('The accuracy of the Perceptron is', acc_perceptron)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(train_X, train_Y)
pred_sgd = sgd.predict(test_X)
acc_sgd = round(metrics.accuracy_score(pred_sgd, test_Y)*100, 2)
print('The accuracy of the SGD is', acc_sgd)

# Model evaluation
models = pd.DataFrame({'Model': ['Logistic Regression', 
                                  'Support Vector Machines', 
                                  'Linear SVC', 
                                  'Random Forest', 
                                  'KNN', 
                                  'Naive Bayes', 
                                  'Decision Tree', 
                                  'Perceptron', 
                                  'Stochastic Gradient Decent'], 
                        'Score': [acc_log, 
                                  acc_rbf, 
                                  acc_linear, 
                                  acc_random_forest, 
                                  acc_knn, 
                                  acc_gaussian, 
                                  acc_decision_tree, 
                                  acc_perceptron, 
                                  acc_sgd]})
                                 
print(models.sort_values(by='Score', ascending=False))


#%% Save & Restore

with open('Customer-Churn_Model.pickle', 'wb') as f:
  pickle.dump(acc_random_forest, f)
print('Saving done')

end_time = time.time()
print('Running time: %.2f s' % (end_time - start_time))