import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="WFO vs WFH Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏢 WFO vs WFH Classifier")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page:", 
                    ["📊 Data Overview", 
                     "🔍 Exploratory Analysis", 
                     "🤖 Model Training", 
                     "🎯 Predictions"])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("WFH_WFO_dataset.csv")
    df = df.drop(columns=["ID", "Name"])
    return df

# Encode categorical columns
@st.cache_data
def encode_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    return df_encoded, label_encoders

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    return model

# Load and prepare data
df = load_data()
df_encoded, label_encoders = encode_data(df)

# Prepare features and target
X = df_encoded.drop("Target", axis=1)
y = df_encoded["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = train_model(X_train_scaled, y_train)

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "📊 Data Overview":
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isna().sum().sum())
    with col4:
        st.metric("Data Types", df.dtypes.nunique())
    
    st.subheader("Dataset Head")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Missing Values:**")
        st.write(df.isna().sum())
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ==================== PAGE 2: EXPLORATORY ANALYSIS ====================
elif page == "🔍 Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Variable Distribution")
        target_counts = df_encoded['Target'].value_counts()
        fig = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            labels={'x': 'Target', 'y': 'Count'},
            title='WFH vs WFO Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Target Variable (Pie Chart)")
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title='Proportion of WFH vs WFO'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Box Plot Analysis")
    fig, ax = plt.subplots(figsize=(15, 8))
    df.boxplot(ax=ax)
    plt.title("Box Plot of All Features")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df_encoded.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)

# ==================== PAGE 3: MODEL TRAINING ====================
elif page == "🤖 Model Training":
    st.header("SVM Model Training & Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        st.write("- **Algorithm:** Support Vector Machine (SVM)")
        st.write("- **Kernel:** RBF (Radial Basis Function)")
        st.write("- **Train Size:** 80%")
        st.write("- **Test Size:** 20%")
        st.write("- **Scaler:** StandardScaler")
    
    with col2:
        st.subheader("Dataset Split")
        st.write(f"- **Training Samples:** {len(X_train)}")
        st.write(f"- **Testing Samples:** {len(X_test)}")
        st.write(f"- **Total Features:** {X.shape[1]}")
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Accuracy", f"{train_accuracy:.4f}", f"{train_accuracy*100:.2f}%")
    
    with col2:
        st.metric("Testing Accuracy", f"{test_accuracy:.4f}", f"{test_accuracy*100:.2f}%")
    
    st.subheader("Classification Report")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_test_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['WFH', 'WFO'], yticklabels=['WFH', 'WFO'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig)
    
    # Print classification report
    st.subheader("Detailed Classification Report")
    st.text(classification_report(y_test, y_test_pred))

# ==================== PAGE 4: PREDICTIONS ====================
elif page == "🎯 Predictions":
    st.header("Make Predictions")
    
    st.write("Enter feature values to predict WFH or WFO:")
    
    # Create input columns for each feature
    feature_names = X.columns.tolist()
    
    st.subheader("Feature Inputs")
    
    # Create a form for user inputs
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        user_input = {}
        for idx, feature in enumerate(feature_names):
            if idx % 3 == 0:
                with col1:
                    user_input[feature] = st.number_input(
                        f"{feature}",
                        value=float(X[feature].mean()),
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        step=0.1
                    )
            elif idx % 3 == 1:
                with col2:
                    user_input[feature] = st.number_input(
                        f"{feature}",
                        value=float(X[feature].mean()),
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        step=0.1
                    )
            else:
                with col3:
                    user_input[feature] = st.number_input(
                        f"{feature}",
                        value=float(X[feature].mean()),
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        step=0.1
                    )
        
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
    
    if submitted:
        # Prepare input data
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        
        # Make prediction
        prediction = model.predict(user_scaled)[0]
        probabilities = model.predict_proba(user_scaled)[0]
        
        st.subheader("Prediction Result")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 0:
                st.success("🏠 **Prediction: WFH (Work From Home)**")
            else:
                st.success("🏢 **Prediction: WFO (Work From Office)**")
        
        with result_col2:
            st.info(f"**Confidence:** {max(probabilities)*100:.2f}%")
        
        st.subheader("Probability Distribution")
        prob_data = pd.DataFrame({
            'Category': ['WFH', 'WFO'],
            'Probability': probabilities
        })
        
        fig = px.bar(prob_data, x='Category', y='Probability', 
                     title='Prediction Probabilities',
                     color='Probability', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)