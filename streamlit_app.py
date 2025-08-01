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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Predict passenger survival on the Titanic using machine learning models
    </p>
</div>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    # Using the famous Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def preprocess_data(df):
    # Create a copy for preprocessing
    data = df.copy()
    
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Create new features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Simplify titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna('Rare', inplace=True)
    
    # Age groups
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                             labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare groups
    data['FareGroup'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    return data

# Load data
df_raw = load_data()
df = preprocess_data(df_raw)

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a section:", 
                           ["📊 Data Overview", "🔍 Data Analysis", "🤖 Model Training", "🎯 Prediction"])

if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        st.metric("Survivors", df['Survived'].sum())
    with col3:
        st.metric("Survival Rate", f"{df['Survived'].mean():.1%}")
    with col4:
        st.metric("Features", len(df.columns))
    
    # Display sample data
    st.subheader("📋 Sample Data")
    if st.checkbox("Show raw data"):
        st.dataframe(df_raw.head(10), use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.subheader("🔍 Missing Values")
        missing_data = df_raw.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df_raw) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

elif page == "🔍 Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    # Survival analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(df['Survived'].value_counts().reset_index(), 
                     x='index', y='Survived', 
                     title="Survival Distribution",
                     labels={'index': 'Survived', 'Survived': 'Count'},
                     color='index',
                     color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'})
        fig1.update_xaxis(tickvals=[0, 1], ticktext=['Died', 'Survived'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        survival_by_class = df.groupby(['Pclass', 'Survived']).size().unstack()
        fig2 = px.bar(survival_by_class.reset_index(), 
                     x='Pclass', y=[0, 1],
                     title="Survival by Passenger Class",
                     labels={'value': 'Count', 'variable': 'Survived'},
                     color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Interactive analysis
    st.subheader("🎛️ Interactive Analysis")
    analysis_type = st.selectbox("Choose analysis type:", 
                                ["Age Distribution", "Fare Analysis", "Family Size Impact", "Embarkation Analysis"])
    
    if analysis_type == "Age Distribution":
        fig = px.histogram(df, x='Age', color='Survived', nbins=30,
                          title="Age Distribution by Survival",
                          color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'})
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Fare Analysis":
        fig = px.box(df, x='Survived', y='Fare', 
                    title="Fare Distribution by Survival")
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Family Size Impact":
        family_survival = df.groupby('FamilySize')['Survived'].agg(['count', 'sum', 'mean']).reset_index()
        family_survival['survival_rate'] = family_survival['mean']
        
        fig = px.bar(family_survival, x='FamilySize', y='survival_rate',
                    title="Survival Rate by Family Size",
                    labels={'survival_rate': 'Survival Rate'})
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Embarkation Analysis":
        embark_survival = df.groupby(['Embarked', 'Survived']).size().unstack()
        fig = px.bar(embark_survival.reset_index(), 
                    x='Embarked', y=[0, 1],
                    title="Survival by Embarkation Port",
                    labels={'value': 'Count', 'variable': 'Survived'},
                    color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   title="Feature Correlation Matrix",
                   color_continuous_scale='RdBu_r',
                   aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    # Prepare data for modeling
    @st.cache_data
    def prepare_model_data(df):
        # Select features for modeling
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                   'FamilySize', 'IsAlone', 'Title']
        
        X = df[features].copy()
        y = df['Survived']
        
        # Encode categorical variables
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        le_title = LabelEncoder()
        
        X['Sex'] = le_sex.fit_transform(X['Sex'])
        X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
        X['Title'] = le_title.fit_transform(X['Title'])
        
        return X, y, le_sex, le_embarked, le_title
    
    X, y, le_sex, le_embarked, le_title = prepare_model_data(df)
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 100, 42)
    
    with col2:
        # RandomForest hyperparameters
        n_estimators = st.slider("Random Forest - Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Random Forest - Max Depth", 3, 20, 10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'SVM': SVC(probability=True, random_state=random_state)
    }
    
    # Train and evaluate models
    if st.button("🚀 Train Models", type="primary"):
        results = []
        trained_models = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Training {name}...')
            
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            results.append({
                'Model': name,
                'Train Accuracy': f"{train_acc:.3f}",
                'Test Accuracy': f"{test_acc:.3f}",
                'Overfitting': f"{train_acc - test_acc:.3f}"
            })
            
            trained_models[name] = model
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text('Training completed!')
        
        # Display results
        st.subheader("📊 Model Performance")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Best model
        best_model_name = results_df.loc[results_df['Test Accuracy'].astype(float).idxmax(), 'Model']
        st.success(f"🏆 Best Model: {best_model_name}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in trained_models:
            st.subheader("🎯 Feature Importance (Random Forest)")
            rf_model = trained_models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        
        # Store models in session state
        st.session_state.trained_models = trained_models
        st.session_state.scaler = scaler
        st.session_state.encoders = (le_sex, le_embarked, le_title)
        st.session_state.feature_names = X.columns.tolist()

elif page == "🎯 Prediction":
    st.header("🎯 Survival Prediction")
    
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' section.")
        st.stop()
    
    # Input form
    st.subheader("👤 Passenger Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)
        
    with col2:
        sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
        parch = st.number_input("Parents/Children", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 50.0)
        
    with col3:
        embarked = st.selectbox("Embarkation Port", ["C", "Q", "S"], index=2)
        title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])
    
    # Calculate derived features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'Title': [title]
    })
    
    # Encode input data
    le_sex, le_embarked, le_title = st.session_state.encoders
    input_encoded = input_data.copy()
    input_encoded['Sex'] = le_sex.transform([sex])[0]
    input_encoded['Embarked'] = le_embarked.transform([embarked])[0]
    input_encoded['Title'] = le_title.transform([title])[0]
    
    # Make predictions
    st.subheader("🔮 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Survival Probabilities")
        for name, model in st.session_state.trained_models.items():
            if name in ['Logistic Regression', 'SVM']:
                input_scaled = st.session_state.scaler.transform(input_encoded)
                proba = model.predict_proba(input_scaled)[0]
                prediction = model.predict(input_scaled)[0]
            else:
                proba = model.predict_proba(input_encoded)[0]
                prediction = model.predict(input_encoded)[0]
            
            survival_prob = proba[1]
            
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = survival_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"{name}"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Final Predictions")
        predictions_summary = []
        
        for name, model in st.session_state.trained_models.items():
            if name in ['Logistic Regression', 'SVM']:
                input_scaled = st.session_state.scaler.transform(input_encoded)
                proba = model.predict_proba(input_scaled)[0]
                prediction = model.predict(input_scaled)[0]
            else:
                proba = model.predict_proba(input_encoded)[0]
                prediction = model.predict(input_encoded)[0]
            
            predictions_summary.append({
                'Model': name,
                'Prediction': '✅ Survived' if prediction == 1 else '❌ Did not survive',
                'Confidence': f"{max(proba):.1%}"
            })
        
        pred_df = pd.DataFrame(predictions_summary)
        st.dataframe(pred_df, use_container_width=True)
        
        # Ensemble prediction
        ensemble_proba = np.mean([
            model.predict_proba(st.session_state.scaler.transform(input_encoded) if name in ['Logistic Regression', 'SVM'] else input_encoded)[0][1]
            for name, model in st.session_state.trained_models.items()
        ])
        
        st.markdown("### 🏆 Ensemble Prediction")
        if ensemble_proba > 0.5:
            st.success(f"✅ **SURVIVED** (Confidence: {ensemble_proba:.1%})")
        else:
            st.error(f"❌ **DID NOT SURVIVE** (Confidence: {1-ensemble_proba:.1%})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🚢 Titanic Survival Predictor | Built with Streamlit & Machine Learning</p>
    <p>Data source: <a href="https://www.kaggle.com/c/titanic">Kaggle Titanic Dataset</a></p>
</div>
""", unsafe_allow_html=True)



