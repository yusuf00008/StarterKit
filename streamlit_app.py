'''
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  
import category_encoders as ce
import plotly.express as px

st.set_page_config(page_title="🐧 Penguin Classifier", layout="wide")  

st.title('🐧 Penguin Classifier - Обучение и предсказание')
st.write('## Работа с датасетом пингвинов')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

st.subheader('Случайные 10 срок')
st.dataframe(df.sample(10), use_container_width = True)


st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="species", color="island", barmode="group", title="Распределение видов по островам")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species", title="Длина клюва vs Длина крыла")
    st.plotly_chart(fig2, use_container_width=True)


X = df.drop(["species"], axis = 1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

encoder = ce.TargetEncoder(cols=["island", "sex"])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
    acc_test = accuracy_score(y_test, model.predict(X_test_encoded))
    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.write("### 🧪 Сравнение моделей по точности")
st.table(pd.DataFrame(results))


st.sidebar.header("🔍 Предсказание по параметрам")
island_input = st.sidebar.selectbox("Остров", df['island'].unique())
sex_input = st.sidebar.selectbox("Пол", df['sex'].unique())
bill_length = st.sidebar.slider("Длина клюва (мм)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Глубина клюва (мм)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Длина крыла (мм)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider("Масса тела (г)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))


user_input = pd.DataFrame({
    'island': island_input,
    'sex': sex_input,
    'bill_length_mm': bill_length,
    'bill_depth_mm': bill_depth,
    'flipper_length_mm': flipper_length,
    'body_mass_g': body_mass
}, index=[0])


user_encoded = encoder.transform(user_input)
for col in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    user_encoded[col] = user_input[col].values

user_encoded = user_encoded[X_train_encoded.columns]


st.sidebar.subheader("🔮 Результаты предсказания")
for name, model in models.items():
    pred = model.predict(user_encoded)[0]  
    proba = model.predict_proba(user_encoded)[0]
    st.sidebar.markdown(f"**{name}: {pred}**")
    proba_df = pd.DataFrame({'Вид': model.classes_, 'Вероятность': proba})
    st.sidebar.dataframe(proba_df.set_index("Вид"), use_container_width=True)
'''

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Page setup
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("🚢 Titanic Survival Predictor")
st.write("Predict if a passenger would survive the Titanic disaster using RandomForest!")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

# Simple data preprocessing
@st.cache_data
def preprocess_data(df):
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    return data

# Load and preprocess data
df_raw = load_data()
df = preprocess_data(df_raw)

# Show basic info about dataset
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Passengers", len(df))
with col2:
    st.metric("Survivors", df['Survived'].sum())
with col3:
    st.metric("Survival Rate", f"{df['Survived'].mean():.1%}")

# Show sample data
st.subheader("Sample Data")
st.dataframe(df.head(10))

# Data visualization - IMPROVEMENT 1 (+10 points)
st.header("📈 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    # Survival by class
    fig1 = px.bar(df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count'), 
                  x='Pclass', y='Count', color='Survived',
                  title="Survival by Passenger Class")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Survival by gender
    fig2 = px.bar(df.groupby(['Sex', 'Survived']).size().reset_index(name='Count'), 
                  x='Sex', y='Count', color='Survived',
                  title="Survival by Gender")
    st.plotly_chart(fig2, use_container_width=True)

# Age distribution
fig3 = px.histogram(df, x='Age', color='Survived', 
                   title="Age Distribution by Survival")
st.plotly_chart(fig3, use_container_width=True)

# IMPROVEMENT 2: Hyperparameter tuning (+10 points)
st.header("🤖 Model Configuration")
st.write("Adjust RandomForest parameters:")

col1, col2 = st.columns(2)
with col1:
    n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
    max_depth = st.slider("Maximum Depth", 3, 20, 10)
with col2:
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

# Prepare data for modeling
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
X = df[features].copy()
y = df['Survived']

# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

X['Sex'] = le_sex.fit_transform(X['Sex'])
X['Embarked'] = le_embarked.fit_transform(X['Embarked'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model with user parameters
st.subheader("Training RandomForest Model...")

rf_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)

# Calculate accuracy
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

# Show results
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{train_acc:.3f}")
with col2:
    st.metric("Test Accuracy", f"{test_acc:.3f}")

# Feature importance
st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig4 = px.bar(importance_df, x='Importance', y='Feature', 
              orientation='h', title="Which features are most important?")
st.plotly_chart(fig4, use_container_width=True)

# IMPROVEMENT 3: Interactive prediction interface (+10 points)
st.header("🎯 Make a Prediction")
st.write("Enter passenger details to predict survival:")

# Input form with better interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    input_pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2, 
                               help="1=First Class, 2=Second Class, 3=Third Class")
    input_sex = st.selectbox("Sex", ["male", "female"])
    input_age = st.slider("Age", 0, 100, 30)

with col2:
    st.subheader("Travel Details")
    input_sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    input_parch = st.number_input("Parents/Children aboard", 0, 10, 0)
    input_fare = st.slider("Fare paid", 0.0, 500.0, 50.0)
    input_embarked = st.selectbox("Embarkation Port", 
                                 ["C", "Q", "S"], 
                                 index=2,
                                 help="C=Cherbourg, Q=Queenstown, S=Southampton")

# Calculate family size
input_family_size = input_sibsp + input_parch + 1

# Show family info
st.info(f"Family Size: {input_family_size} people")

# Create input dataframe
input_data = pd.DataFrame({
    'Pclass': [input_pclass],
    'Sex': [le_sex.transform([input_sex])[0]],
    'Age': [input_age],
    'SibSp': [input_sibsp],
    'Parch': [input_parch],
    'Fare': [input_fare],
    'Embarked': [le_embarked.transform([input_embarked])[0]],
    'FamilySize': [input_family_size]
})

# Make prediction
prediction = rf_model.predict(input_data)[0]
probability = rf_model.predict_proba(input_data)[0]

# Show prediction with nice formatting
st.subheader("Prediction Result")

if prediction == 1:
    st.success(f"✅ **SURVIVED** - Survival Probability: {probability[1]:.1%}")
    st.balloons()
else:
    st.error(f"❌ **DID NOT SURVIVE** - Death Probability: {probability[0]:.1%}")

# Show probability breakdown
prob_df = pd.DataFrame({
    'Outcome': ['Did not survive', 'Survived'],
    'Probability': [f"{probability[0]:.1%}", f"{probability[1]:.1%}"]
})
st.dataframe(prob_df, use_container_width=True)

# Additional insights
st.write("---")
st.subheader("💡 Insights from the Model")
st.write("Based on the Titanic dataset analysis:")
st.write("- **Women** had much higher survival rates than men")
st.write("- **First class** passengers were more likely to survive")
st.write("- **Younger passengers** generally had better survival chances")
st.write("- **Smaller families** often had better survival rates")

# Model info
st.write("---")
st.write("**Model Details:**")
st.write(f"- Algorithm: Random Forest with {n_estimators} trees")
st.write(f"- Training Accuracy: {train_acc:.1%}")
st.write(f"- Test Accuracy: {test_acc:.1%}")
st.write("- Dataset: Titanic passenger data from Kaggle")




