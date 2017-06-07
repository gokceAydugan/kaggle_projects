import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# read the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
data = [train_data, test_data]

# explore the training data
print(train_data.columns.values)
print(test_data.columns.values)
# Survived is prediction value
print(train_data.head())

# Categorical: Sex, Embarked, Survived
# Ordinal: PClass
# Continuous: Age, PassengerID, Fare
# Discrete: SibSp, Parch
# Mixed: Ticket
print(train_data['Cabin'].unique())
# Alphanumeric: Cabin

# Missing Value Detection
print(pd.isnull(train_data).sum() > 0)
# Age, Cabin, Embarked
print(pd.isnull(test_data).sum() > 0)
# Age, Cabin, Fare

print(train_data.info())
# float || int : PassengerId, Survived, Pclass, Age,SibSp, Parch, Fare
# Object : Name, Sex, Ticket, Cabin, Embarked
# Includes null: Age, Cabin, Embarked
# Total Samples : 891, 40% (Actual is 2224)
print(train_data["Survived"].sum())
# 38%  is survived (Actual is 32%)
print(len(train_data[train_data["Parch"] == 0]) * 100 / len(train_data["Parch"]))
# 76% travelled without Parch
print(len(train_data[train_data["SibSp"] == 0]) * 100 / len(train_data["SibSp"]))
# 68% travelled without Siblings and Spouse

print(test_data.info())
# Includes null: Age, Cabin, Fare

uniques = []
for column in train_data.columns:
    u = len(train_data[column].unique())
    print(column + str(" is unique: ") + str(u))
    if u < 10:
        uniques.append(column)
# Examine with graph: Embarked, Sibsp, Parch, Sex

print(train_data["Sex"].unique())
# male or female
print(len(train_data[train_data["Sex"] == "male"]) * 100 / len(train_data["Sex"]))
# ~65% is male

for column in train_data.columns:
    unique = True if len(train_data[column].unique()) / len(train_data[column]) == 1 else False
    print(column + str(" is unique: ") + str(unique))
# only Name and PassengerId is unique

# Detect Correlations
pred = 'Survived'
uniques.remove(pred)
print(uniques)
for c in uniques:
    print(train_data[[c, pred]].groupby([c], as_index=False).mean().sort_values(by=pred, ascending=False))
# Embarked : C > Q > S (*)
# Parch : 3 > 1 > 2 > 0 > 5 > 4 > 6
# SibSp : 1 > 2 > 0 > 3 > 4 > 5 > 8 (*)
# Sex : F > M (*)
# Pclass : 1 > 2 > 3 (*)

g = sns.FacetGrid(train_data, col=pred)
g.map(plt.hist, 'Age', bins=20)
# Age 20-40 : Mostly died
# Age 60-80 : died
# Infants : mostly survived
# Elder : people survived
# Age is correlated (*)

for c in uniques:
    grid = sns.FacetGrid(train_data, col=pred,
                         row=c, size=2.2,
                         aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# Embarked ~ Age, doesn't give extra information
# Adult male mostly died
# Pclass 3 Age 20-40 mostly died
# Pclass 1 most likely to survive

# Age, Pclass, Sex are important
grid = sns.FacetGrid(train_data, col=pred, row='Sex')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# Boy infants most likely to survive
# Male and Age 20-40 : mostly died
# Female mostly survived
female = train_data[train_data["Sex"] == "female"]
female_s = female[female["Survived"] == 1]
print(len(female_s) / len(female))
# 74%
grid = sns.FacetGrid(train_data, col=pred, row='Sex')
grid.map(plt.hist, 'Pclass', alpha=.5)
grid.add_legend();
# Pclass = 1 & Female mostly survived

grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# Embarked = Q, Pclass = 3, Male most likely to survive
# Higher Pclass, higher chance to survive for females
# Add Embarked to the model(?)
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# Higher fare paying passengers had better survival.
# Add Fare to the model

# Correlating categorical and numerical features
print("Before", train_data.shape, test_data.shape, data[0].shape, data[1].shape)

# Drop : Ticket, Cabin, PassengerId
train_data = train_data.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
data = [train_data, test_data]
print("After", train_data.shape, test_data.shape, data[0].shape, data[1].shape)

for d in data:
    d['Title'] = d.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])

# replace titles with more common names or classify as rare
for d in data:
    d['Title'] = d['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')

print(train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Categorical titles to ordinal
t_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for d in data:
    d['Title'] = d['Title'].map(t_map)
    d['Title'] = d['Title'].fillna(0)

print(train_data.head())

# Drop : Name
train_data = train_data.drop(['Name'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
data = [train_data, test_data]
print("After", train_data.shape, test_data.shape, data[0].shape, data[1].shape)

# Categorical Sex Mapping to int 0 & 1
for d in data:
    d['Sex'] = d['Sex'].map({'female': 1, 'male': 0}).astype(int)
print(train_data.head())

# Complete numerical continous features
grid = sns.FacetGrid(train_data, row="Pclass", col="Sex")
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# Fill the null ages with the mean of the samples with same Sex and Pclass
ages = np.zeros((2, 3))
for d in data:
    for i in range(0, 2):
        for j in range(0, 3):
            guess = d[(d['Sex'] == i) & (d['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess.mean()
            ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            d.loc[(d.Age.isnull()) & (d.Sex == i) & (d.Pclass == j + 1), 'Age'] = ages[i, j]
    d['Age'] = d['Age'].astype(int)

print(train_data.head())

# Examine the survival rate among Age bands
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
print(train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                                  ascending=True))

# Replace Age with AgeBand
for d in data:
    d.loc[d['Age'] <= 16, 'Age'] = 0
    d.loc[(d['Age'] > 16) & (d['Age'] <= 32), 'Age'] = 1
    d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age'] = 2
    d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age'] = 3
    d.loc[d['Age'] > 64, 'Age']

print(train_data.head())
# Remove AgeBand feature
train_data = train_data.drop(['AgeBand'], axis=1)
data = [train_data, test_data]

for d in data:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

print(train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                        ascending=False))

for d in data:
    d['IsAlone'] = 0
    d.loc[d['FamilySize'] == 1, 'IsAlone'] = 1

print(train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
# Alone people died mostly

# Drop Parch, SibSp, FamilySize
train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
data = [train_data, test_data]

print(train_data.shape)

# Complete categorical values
fred_port = train_data.Embarked.dropna().mode()[0]
for d in data:
    d['Embarked'] = d['Embarked'].fillna(fred_port)

print(train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# map S -> 0, C -> 1, Q -> 2
for d in data:
    d['Embarked'] = d['Embarked'].map({'S':0, 'C':1, 'Q':2})

# Also complete Fare in Test data
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

# PREDICTION
X_train = train_data.drop('Survived', axis=1)
Y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis=1).copy()

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_logreg = round(model.score(X_train,Y_train)*100,2)
print("Logistic Regression: ", acc_logreg)

coeff_df = pd.DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(model.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))
# SEX: highest (+) coefficient : Higher sex higher survival ratio
# Pclass : lowest (-) coefficient : Lower Pclass, higher survival ratio

# SVM
model = SVC()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_svm = round(model.score(X_train, Y_train)*100,2)
print("SVM: ", acc_svm)

# Perceptron
model = Perceptron()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_perceptron = round(model.score(X_train,Y_train)*100,2)
print("Perceptron: ", acc_perceptron)

# Stochastic Gradient Descent
model = SGDClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_sgd = round(model.score(X_train,Y_train)*100,2)
print("Stochastic Gradient Descent: ", acc_sgd)

# Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_rf = round(model.score(X_train,Y_train)*100,2)
print("Random Forest (100): ", acc_rf)

model = RandomForestClassifier(n_estimators=5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Random Forest 5: ", round(model.score(X_train,Y_train)*100,2))

model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("Random Forest 1000: ", round(model.score(X_train,Y_train)*100,2))

# n = 100 gives same results with n=1000 bur much faster. n=5 is faster than n=100 but gives worse results

# Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_dt = round(model.score(X_train,Y_train)*100,2)
print("Random Forest (100): ", acc_dt)

# EVALUATION
models = pd.DataFrame({'Model': ['Support Vector Machines', 'Logistic Regression',
              'Random Forest', 'Perceptron',
              'Stochastic Gradient Decent', 'Decision Tree'],
    'Score': [acc_svm, acc_logreg,
              acc_rf, acc_perceptron,
              acc_sgd, acc_dt]})
print(models.sort_values(by='Score', ascending=False))

# since the decision tree and random forest gave same result, we submet the last run one directly
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred})
submission.to_csv("predictions.csv",index=False)