import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")


# Data Preprocessing
def data_preprocess(df, is_training):

    scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # missing_count = df['ColumnName'].isna().sum()
    # unique_count = df['ColumnName'].nunique()
    # value_counts = df['ColumnName'].value_counts()

    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)

    encoded_planet = pd.get_dummies(df['HomePlanet'], dummy_na=True, prefix='Planet')
    df = pd.concat([df, encoded_planet], axis=1)
    df.drop('HomePlanet', axis=1, inplace=True)

    encoded_cryosleep = pd.get_dummies(df['CryoSleep'], dummy_na=True, prefix='CryoSleep')
    df = pd.concat([df, encoded_cryosleep], axis=1)
    df.drop('CryoSleep', axis=1, inplace=True)

    # Cabin
    split_cabin = df['Cabin'].str.split('/', expand=True)
    split_cabin.columns = ['Deck', 'Number', 'Side']
    df = pd.concat([df, split_cabin], axis=1)
    df.drop('Cabin', axis=1, inplace=True)

    encoded_deck = pd.get_dummies(df['Deck'], dummy_na=True, prefix='Deck')
    df = pd.concat([df, encoded_deck], axis=1)
    df.drop('Deck', axis=1, inplace=True)

    df.drop('Number', axis=1, inplace=True)

    encoded_side = pd.get_dummies(df['Side'], dummy_na=True, prefix='Side')
    df = pd.concat([df, encoded_side], axis=1)
    df.drop('Side', axis=1, inplace=True)

    # Destination
    encoded_destination = pd.get_dummies(df['Destination'], dummy_na=True, prefix='Destination')
    df = pd.concat([df, encoded_destination], axis=1)
    df.drop('Destination', axis=1, inplace=True)

    # Age
    mean_age = df['Age'].mean()
    df['Age'].fillna(mean_age, inplace=True)
    df['ScaledAge'] = scaler.fit_transform(df[['Age']])
    df.drop('Age', axis=1, inplace=True)

    df['VIP'] = df['VIP'].map({True: 1, False: 0})
    df['VIP'].fillna(0, inplace=True)

    df['RoomService'].fillna(0, inplace=True)
    df['ScaledRoomService'] = standard_scaler.fit_transform(df[['RoomService']])
    df.drop('RoomService', axis=1, inplace=True)

    df['FoodCourt'].fillna(0, inplace=True)
    df['ScaledFoodCourt'] = standard_scaler.fit_transform(df[['FoodCourt']])
    df.drop('FoodCourt', axis=1, inplace=True)

    df['ShoppingMall'].fillna(0, inplace=True)
    df['ScaledShoppingMall'] = standard_scaler.fit_transform(df[['ShoppingMall']])
    df.drop('ShoppingMall', axis=1, inplace=True)

    df['Spa'].fillna(0, inplace=True)
    df['ScaledSpa'] = standard_scaler.fit_transform(df[['Spa']])
    df.drop('Spa', axis=1, inplace=True)

    df['VRDeck'].fillna(0, inplace=True)
    df['ScaledVRDeck'] = standard_scaler.fit_transform(df[['VRDeck']])
    df.drop('VRDeck', axis=1, inplace=True)
    if is_training:
        df['Transported'] = df['Transported'].map({True: 1, False: 0})

    return df


# train_data.to_excel("Processed_Data.xlsx")

train_data = data_preprocess(train_data, True)

# Split Data
labels = train_data['Transported']
features = train_data.drop('Transported', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=0)

# Model Training
model = RandomForestClassifier(n_estimators=80, random_state=42)  # Adjust hyperparameters as needed

model.fit(X_train, y_train)

# Model Test and Evaluation
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

training_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Training Accuracy: {training_accuracy}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Predict
submission_df = pd.DataFrame()
submission_df['PassengerId'] = test_data['PassengerId']
test_data = data_preprocess(test_data, False)
predictions = model.predict(test_data)
submission_df['Transported'] = predictions
submission_df['Transported'] = submission_df['Transported'].map({1: True, 0: False})
submission_df.to_csv("Submission.csv", index=False)