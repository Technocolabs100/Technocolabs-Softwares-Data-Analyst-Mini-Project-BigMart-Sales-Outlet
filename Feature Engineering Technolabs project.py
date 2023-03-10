#!/usr/bin/env python
# coding: utf-8

# In[145]:


import pandas as pd
df=pd.read_csv('Prosper Loan Data.csv')


# In[146]:


df.head()


# In[147]:


df.tail()


# In[148]:


df.describe()


# In[149]:


df.info()


# In[150]:


df.dtypes


# In[151]:


import numpy as np
categorical = df.select_dtypes(include=["object"]).columns.values
df[categorical] = df[categorical].fillna("Unknown")

df.select_dtypes(exclude=[np.number]).isnull().sum()


# In[152]:


df.isnull().sum()*100 / len(df)


# In[153]:


borrower_fees = df["BorrowerAPR"] - df["BorrowerRate"]
borrower_fees.median()


# In[154]:


df["BorrowerAPR"].fillna(df["BorrowerRate"] + borrower_fees.median(), inplace=True)

df["BorrowerAPR"].isnull().sum()


# In[155]:


estimated_loss_from_fees = df["BorrowerRate"] - df["EstimatedEffectiveYield"]
estimated_loss_from_fees.median()


# In[156]:


df["EstimatedEffectiveYield"].fillna(df["BorrowerRate"] - estimated_loss_from_fees.median(), inplace=True)

df["EstimatedEffectiveYield"].isnull().sum()


# In[157]:


df['EstimatedLoss'].fillna(df['EstimatedLoss'].mean(), inplace = True)


# In[158]:


df["EstimatedReturn"].fillna(df["EstimatedEffectiveYield"] - df["EstimatedLoss"], inplace=True)
df["EstimatedReturn"].isnull().sum()


# In[159]:


df["ProsperRating (numeric)"].fillna(df["ProsperRating (numeric)"].median(), inplace=True)
df["ProsperScore"].fillna(df["ProsperScore"].median(), inplace=True)


df["ProsperRating (numeric)"].isnull().sum(), df["ProsperScore"].isnull().sum()


# In[160]:


df.dropna(subset=["EmploymentStatusDuration", "CreditScoreRangeLower", "FirstRecordedCreditLine", "CurrentCreditLines",
                  "TotalCreditLinespast7years"], inplace=True)


# In[161]:


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[162]:


df.info()


# In[163]:


def null_values() :  
    lst = df.isnull().sum()
    for i in range(len(lst)) :
        if lst[i] != 0 :
            x= lst[i]
            col = df.columns[i]
            y= (x/df.shape[0])*100
            print("Column No : " +str(i) + " -  " + col + " - " + str(x) + " nulls -"+ str(round(y,2)) +" %")
null_values()         


# In[164]:


df_debt_income_null = df[df["DebtToIncomeRatio"].isnull()]
df_debt_income_null[:5]


# In[165]:


df_debt_income_null = df[df["DebtToIncomeRatio"].isnull()]
df_debt_income_null["MonthlyLoanPayment"].isnull().sum(), df_debt_income_null["StatedMonthlyIncome"].isnull().sum()


# In[166]:


df_debt_income_null["IncomeVerifiable"][:10]


# In[167]:


#Calculate DebtToIncomeRatio for unverifiable incomes, adding $1 to account for $0/month incomes
df["DebtToIncomeRatio"].fillna(df["MonthlyLoanPayment"] / (df["StatedMonthlyIncome"] + 1), inplace = True)

df["DebtToIncomeRatio"].isnull().sum()


# In[168]:


df.drop("ScorexChangeAtTimeOfListing", axis=1, inplace=True)


# In[169]:


df.shape


# In[170]:


prosper_vars = ["TotalProsperLoans","TotalProsperPaymentsBilled", "OnTimeProsperPayments", "ProsperPaymentsLessThanOneMonthLate",
                "ProsperPaymentsOneMonthPlusLate", "ProsperPrincipalBorrowed", "ProsperPrincipalOutstanding"]

df[prosper_vars] = df[prosper_vars].fillna(0)


# In[172]:


df.LoanStatus.unique()


# In[173]:


df.info()


# In[174]:


features =['LoanOriginalAmount','ListingCategory (numeric)','BorrowerAPR','BorrowerRate','StatedMonthlyIncome',
          'ProsperScore','Term','BankcardUtilization','DebtToIncomeRatio','MonthlyLoanPayment','TotalTrades','Investors']


# In[175]:


X = df[features]
y = df['LoanStatus']


# In[176]:


y.unique()


# In[177]:


import seaborn as sns
sns.displot(df['BorrowerRate'], kde=True)


# In[178]:


sns.distplot(df['EstimatedLoss'], kde=True)


# In[179]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[180]:


#Multivariate Exploration

g = sns.FacetGrid(data = df, col = 'TotalProsperLoans', height = 3,
                margin_titles = True)
g.map(plt.scatter, 'EstimatedReturn', 'BorrowerRate');


# In[181]:


g = sns.FacetGrid(data = df, col = 'LoanStatus', height = 5,
                margin_titles = True)
g.map(plt.scatter, 'EstimatedLoss', 'BorrowerRate');


# In[182]:


#bivariate analysis 
sns.countplot(data = df, x = 'LoanStatus', hue = 'Term')
plt.xticks(rotation = 90)
plt.xlabel('LoanStatus')
plt.ylabel('Count')
plt.title(' Loan Statuts Vs Term');


# In[183]:


from xgboost import XGBRegressor


# In[184]:


df.info()


# In[185]:


df.columns


# In[186]:


df['LoanStatus'] = df['LoanStatus'].apply(lambda x: x.split(" ")[0]).astype(str) 


# In[187]:


df.drop(["CreditGrade", "LenderYield", "EstimatedEffectiveYield", "EstimatedLoss", "EstimatedReturn",
                 "ProsperRating (Alpha)", "Occupation", "CurrentlyInGroup", "GroupKey", "IncomeRange", "PercentFunded"], axis=1,
                inplace=True)


# In[188]:


df=df.iloc[:83955]
df


# In[189]:


df['LoanStatus'] = df['LoanStatus'].apply(lambda x: x.split(" ")[0]).astype(str) 


# In[190]:


df.LoanStatus.unique()


# In[191]:


Status_mapping = {
           'Current': 1,
           'Completed': 1,
           'Past': 0,
            'Defaulted': 0,
            'Chargedoff': 0,
            'FinalPaymentInProgress': 0}

df['LoanStatus'] = df['LoanStatus'].map(Status_mapping)


# In[192]:


features =['LoanOriginalAmount','ListingCategory (numeric)','BorrowerAPR','BorrowerRate','StatedMonthlyIncome',
          'ProsperScore','Term','BankcardUtilization','DebtToIncomeRatio','MonthlyLoanPayment','TotalTrades','Investors']


# In[193]:


X = df[features]
y = df['LoanStatus']


# In[194]:


# Utility functions
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# In[211]:


from sklearn.feature_selection import mutual_info_regression

mi_scores = make_mi_scores(X, y)
plt.figure(dpi=100, figsize=(10, 6))
plot_mi_scores(mi_scores)


# In[212]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[213]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[214]:


df["LoanStatus"].value_counts()


# In[215]:


#drop Unnamed Coulmn 
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[216]:


df.info()


# In[217]:


fig = plt.figure()

ax1 = fig.add_subplot(221)
sns.countplot(df["LoanStatus"])

ax2 = fig.add_subplot(222)
sns.barplot(y=df["LoanStatus"]).set_ylim([0,1])


# In[218]:


df["LoanStatus"].mean(), 1 - df["LoanStatus"].mean()


# In[220]:


df.describe()


# In[221]:


df.isnull().sum()


# In[222]:


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)


# In[223]:


feat_importances = pd.Series(model.feature_importances_ , index = X.columns)
feat_importances.nlargest(20).plot(kind = 'barh')
plt.show()


# In[224]:


from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(X_train_scaled)

X_train_pca10 = pca.transform(X_train_scaled)
X_test_pca10 = pca.transform(X_test_scaled)

pca.explained_variance_ratio_


# In[225]:


pca = PCA(n_components=3)
pca.fit(X_train)

X_train_pca3 = pca.transform(X_train_scaled)
X_test_pca3 = pca.transform(X_test_scaled)

pca.explained_variance_ratio_


# In[226]:


from sklearn.feature_selection import SelectPercentile

X_train_reduce50 = SelectPercentile(percentile=50).fit_transform(X_train_scaled, y_train)
X_test_reduce50 = SelectPercentile(percentile=50).fit_transform(X_test_scaled, y_test)

X_train_reduce10 = SelectPercentile().fit_transform(X_train_scaled, y_train)
X_test_reduce10 = SelectPercentile().fit_transform(X_test_scaled, y_test)

