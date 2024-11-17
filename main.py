import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Display the first few rows
df.head()

# Check for missing values
df.isnull().sum()

# Fill 'Age' with median and 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to too many missing values
df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.show()

# Age distribution
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Survival by gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.show()

# Survival by passenger class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.show()

# Survival by gender and class
sns.catplot(data=df, x='Sex', hue='Survived', col='Pclass', kind='count')
plt.suptitle('Survival by Gender and Class')
plt.show()





