import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

AirBnb = pd.read_csv('Aemf1.csv')

from sklearn.preprocessing import LabelEncoder
city_names = AirBnb['City']
le = LabelEncoder()
city_labels = le.fit_transform(city_names)
AirBnb['City_L'] = city_labels

day_names = AirBnb['Day']
day_labels = le.fit_transform(day_names)
AirBnb['Day_L'] = city_labels

AirBnb['City'] = AirBnb['City'].astype('category')
AirBnb['Day'] = AirBnb['Day'].astype('category')
AirBnb['Room Type'] = AirBnb['Room Type'].astype('category')

AirBnb.to_csv('Aemf1.csv', index=False)

AirBnb.drop(columns=['City','Day','Room Type'])


# Select the features and target variable
X = AirBnb[['Shared Room','Private Room','Person Capacity','Superhost','Multiple Rooms','Business','Cleanliness Rating','Guest Satisfaction','Bedrooms','City Center (km)','Metro Distance (km)','Attraction Index','Normalised Attraction Index','Restraunt Index','Normalised Restraunt Index','City_L','Day_L']]
y = AirBnb['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred = poly_reg.predict(X_test_poly)



print("R²-Wert: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))