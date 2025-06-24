import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Salary_Data.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Education Level'] = label_encoder.fit_transform(data['Education Level'])
data['Job Title'] = label_encoder.fit_transform(data['Job Title'])
data['marital-status'] = label_encoder.fit_transform(data['marital-status'])
data['workclass'] = label_encoder.fit_transform(data['workclass'])

# Define features and target variable
X = data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'marital-status', 'workclass']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
from sklearn.metrics import r2_score
score=r2_score(y_test, y_pred)
print("The accuracy of our model is {}%".format(round(score, 2) *100))


# Visualize the results
plt.scatter(y_test, y_pred,color="Red",s=50, marker="*")
plt.xlabel('Actual Salaries')
plt.ylabel('Predicted Salaries')
plt.title('Actual vs Predicted Salaries')
plt.show()