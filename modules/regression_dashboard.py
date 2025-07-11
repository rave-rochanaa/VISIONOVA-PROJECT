import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_optimize_data(df, sample_size=1.0):
    """Load and optimize data with sampling"""
    if sample_size < 1.0:
        df = df.sample(frac=sample_size, random_state=42)
    return optimize_memory(df)

def optimize_memory(df):
    """Optimize memory usage of dataframe"""
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() < df.shape[0] * 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        elif df[col].dtype in ['int64', 'float64']:
            if df[col].dtype == 'int64':
                if df[col].min() >= 0:
                    if df[col].max() < 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                    elif df[col].max() < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def feature_selection_module_classification(df, numeric_cols, categorical_cols):
    """Feature selection module for classification"""
    st.header("⚙️ Feature Selection for Classification")
    
    method = st.selectbox("Choose Method", [
        "Correlation Analysis",
        "F-Statistic (ANOVA)",
        "Mutual Information",
        "Recursive Feature Elimination (RFE)",
        "Model-Based Selection"
    ])
    
    filtered_categorical_cols = [col for col in categorical_cols if col.lower() != "campaign_id"]
    target = st.selectbox("🎯 Target Variable", filtered_categorical_cols)
    available_features = [col for col in categorical_cols if col != target]
    features = st.multiselect(
        "🧮 Input Features",
        available_features

    )

    if not features:
        st.warning("⚠️ Please select at least one input feature")
        return

    df_clean = df.dropna(subset=[target] + features).copy()
    
    # Encode categorical variables
    label_encoders = {}
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_clean[target])
    
    X = df_clean[features].copy()
    for col in features:
        if col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    if method == "Correlation Analysis":
        corr_matrix = X.corrwith(pd.Series(y))
        st.write("**Correlation with target variable:**")
        fig = px.bar(x=features, y=corr_matrix.values, 
                    labels={'x': 'Features', 'y': 'Correlation'},
                    title="Feature Correlation with Target")
        st.plotly_chart(fig, use_container_width=True)

    elif method == "F-Statistic (ANOVA)":
        f_values, p_values = f_classif(X, y)
        st.write("**F-Statistic values:**")
        fig = px.bar(x=features, y=f_values,
                    labels={'x': 'Features', 'y': 'F-Statistic'},
                    title="ANOVA F-Statistic Scores")
        st.plotly_chart(fig, use_container_width=True)

    elif method == "Mutual Information":
        mi_scores = mutual_info_classif(X, y)
        st.write("**Mutual Information scores:**")
        fig = px.bar(x=features, y=mi_scores,
                    labels={'x': 'Features', 'y': 'Mutual Information'},
                    title="Mutual Information Scores")
        st.plotly_chart(fig, use_container_width=True)

    elif method == "Recursive Feature Elimination (RFE)":
        model = LogisticRegression(max_iter=1000, random_state=42)
        n_features = min(5, len(features))
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit(X, y)
        selected_features = [features[i] for i, selected in enumerate(rfe.support_) if selected]
        st.write("**Selected Features using RFE:**")
        for feat in selected_features:
            st.write(f"✅ {feat}")

    elif method == "Model-Based Selection":
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        st.write("**Feature Importances from Random Forest:**")
        fig = px.bar(x=features, y=importances,
                    labels={'x': 'Features', 'y': 'Importance'},
                    title="Random Forest Feature Importances")
        st.plotly_chart(fig, use_container_width=True)

def create_classification_model(model_type, params):
    """Create classification model based on type and parameters"""
    if model_type == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42
        )
    elif model_type == "Support Vector Machine":
        return SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
            probability=True,
            random_state=42
        )
    elif model_type == "Naive Bayes":
        return GaussianNB()
    elif model_type == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5))
    elif model_type == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42
        )
    elif model_type == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get('max_depth', None),
            criterion=params.get('criterion', 'gini'),
            random_state=42
        )
    return LogisticRegression(max_iter=1000, random_state=42)

def plot_classification_results(y_test, y_pred, y_pred_proba, model, features, target_encoder, target_name):
    """Plot classification results including confusion matrix and ROC curve"""
    
    # Convert encoded labels back to original labels
    y_test_labels = target_encoder.inverse_transform(y_test)
    y_pred_labels = target_encoder.inverse_transform(y_pred)
    class_names = target_encoder.classes_
    
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])
    
    with tab1:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='Blues',
                       x=class_names,
                       y=class_names)
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📊 Detailed Classification Report")
        report = classification_report(y_test, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=3))
    
    with tab2:
        # ROC Curve (for binary or multiclass)
        if y_pred_proba is not None:
            n_classes = len(class_names)
            
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig = px.line(x=fpr, y=tpr, 
                             title=f'ROC Curve (AUC = {roc_auc:.3f})')
                fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                             line=dict(dash='dash', color='red'))
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Multiclass classification
                st.info("ROC curves for multiclass classification")
                fig = go.Figure()
                
                for i in range(n_classes):
                    y_test_binary = (y_test == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{class_names[i]} (AUC = {roc_auc:.3f})'
                    ))
                
                fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                             line=dict(dash='dash', color='red'))
                fig.update_layout(
                    title='ROC Curves - One vs Rest',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve not available for this model type")
    
    with tab3:
        # Feature Importance/Coefficients
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            fig = px.bar(
                x=features, y=importance,
                labels={'x': 'Features', 'y': 'Importance'},
                title="Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            fig = px.bar(
                x=features, y=coefs,
                labels={'x': 'Features', 'y': 'Coefficient'},
                title="Model Coefficients"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type")

def show_classification_interpretation(model, features, target_name, target_encoder):
    """Show model interpretation for classification"""
    with st.expander("📝 Model Interpretation"):
        if hasattr(model, 'coef_'):
            st.subheader("Model Coefficients")
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            coef_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': coefs
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}))
            
            st.write("**Interpretation:**")
            st.write("- Positive coefficients increase the probability of the positive class")
            st.write("- Negative coefficients decrease the probability of the positive class")
            st.write("- Larger absolute values indicate stronger influence")
            
        elif hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importances")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}))
            
            st.write("**Interpretation:**")
            st.write("- Higher values indicate more important features for prediction")
            st.write("- Values represent the relative contribution to the model's decisions")
        else:
            st.info("Model interpretation not available for this algorithm type")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import traceback

def classification_engine1(df):
    """Main classification engine function"""
    st.title("🎯 Advanced Classification Analytics")
    st.markdown("""
    Predict categorical outcomes using various classification models. 
    Features include model comparison, confusion matrices, ROC curves, and prediction explanations.
    """)

    if df.empty:
        st.error("❌ Uploaded data is empty or failed to load. Please check your CSV file.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) == 0:
        st.error("⚠️ No categorical columns found for classification target.")
        return

    with st.expander("📊 Data Overview", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Total Samples", df.shape[0])
        cols[1].metric("Total Features", df.shape[1])
        cols[2].metric("Numeric Features", len(numeric_cols))
        cols[3].metric("Categorical Features", len(categorical_cols))

    with st.sidebar:
        st.header("📋 Data Sampling")
        sample_size = st.slider("Data Sampling (%)", 1, 100, 100) / 100
        if sample_size < 1:
            st.warning(f"Using {sample_size * 100}% random sample")
    
    if sample_size < 1.0:
        with st.spinner(f"Loading data (sampling: {sample_size * 100}%)..."):
            df = load_and_optimize_data(df, sample_size)
    else:
        df = optimize_memory(df)
        
    if df.empty:
        st.error("❌ Data failed to load or is empty after sampling.")
        return

    mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.success(f"Data loaded: {len(df):,} rows | {mem_usage:.1f} MB")

    if st.checkbox("🔍 Enable Feature Selection", value=False):
        feature_selection_module_classification(df, numeric_cols, categorical_cols)
        st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_type = st.selectbox("Algorithm", [
            "Logistic Regression", 
            "Random Forest",
            "Support Vector Machine",
            "Naive Bayes",
            "K-Nearest Neighbors",
            "Gradient Boosting",
            "Decision Tree"
        ])
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        random_state = st.number_input("Random State", 0, 1000, 42)
        st.markdown("---")
        st.subheader("🎛️ Model Parameters")
        model_params = {}
        if model_type == "Random Forest":
            model_params['n_estimators'] = st.number_input("Number of Trees", 10, 500, 100, step=10)
            model_params['max_depth'] = st.selectbox("Max Depth", [None, 5, 10, 15, 20])
        elif model_type == "Support Vector Machine":
            model_params['C'] = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.01)
            model_params['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        elif model_type == "K-Nearest Neighbors":
            model_params['n_neighbors'] = st.slider("Number of Neighbors", 1, 20, 5)
        elif model_type == "Gradient Boosting":
            model_params['n_estimators'] = st.number_input("Number of Estimators", 50, 500, 100, step=25)
            model_params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        elif model_type == "Decision Tree":
            model_params['max_depth'] = st.selectbox("Max Depth", [None, 5, 10, 15, 20])
            model_params['criterion'] = st.selectbox("Criterion", ["gini", "entropy"])

    col1, col2 = st.columns(2)
    with col1:
        filtered_categorical_cols = [col for col in categorical_cols if col.lower() != "campaign_id"]
        target = st.selectbox("🎯 Target Variable", filtered_categorical_cols)

    with col2:
        all_features = [col for col in numeric_cols + categorical_cols if col != target]
        features = st.multiselect("🧮 Input Features", 
                             all_features,
                             default=all_features[:3] if len(all_features) >= 3 else all_features)

    if not features:
        st.warning("⚠️ Please select at least one input feature")
        return

    if 'trained_classifier' not in st.session_state:
        st.session_state.trained_classifier = None
        st.session_state.clf_features = None
        st.session_state.clf_target_name = None
        st.session_state.clf_model_type = None
        st.session_state.clf_label_encoders = None
        st.session_state.clf_feature_names = None
        st.session_state.clf_df_clean = None
        st.session_state.clf_X_processed = None

    if st.button("🚀 Train & Evaluate Model", type="primary"):
        progress_bar = st.progress(0, text="Preparing data...")
        
        try:
            df_clean = df.dropna(subset=[target] + features).copy()
            if df_clean.empty:
                st.error("❌ No valid data remaining after cleaning missing values.")
                return
            
            progress_bar.progress(10, text="Encoding variables...")
            label_encoders = {}
            le_target = LabelEncoder()
            y = le_target.fit_transform(df_clean[target])
            label_encoders['target'] = le_target
            
            n_classes = len(le_target.classes_)
            st.info(f"🏷️ Target classes ({n_classes}): {list(le_target.classes_)}")

            X = df_clean[features].copy()
            scaler = StandardScaler()
            numeric_features = [col for col in features if col in numeric_cols]
            categorical_features = [col for col in features if col in categorical_cols]
            
            if numeric_features:
                X[numeric_features] = scaler.fit_transform(X[numeric_features])
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

            progress_bar.progress(30, text="Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size/100, 
                random_state=random_state,
                stratify=y if n_classes > 1 else None
            )

            progress_bar.progress(50, text="Training model...")
            model = create_classification_model(model_type, model_params)
            model.fit(X_train, y_train)

            progress_bar.progress(70, text="Making predictions...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            progress_bar.progress(90, text="Calculating metrics...")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            try:
                cv_scores = cross_val_score(model, X, y, cv=min(5, n_classes), scoring='accuracy')
            except:
                cv_scores = [accuracy]

            progress_bar.progress(100, text="Complete!")

            st.session_state.trained_classifier = model
            st.session_state.clf_features = features
            st.session_state.clf_target_name = target
            st.session_state.clf_model_type = model_type
            st.session_state.clf_label_encoders = label_encoders
            st.session_state.clf_feature_names = features
            st.session_state.clf_df_clean = df_clean
            st.session_state.clf_X_processed = X
            st.session_state.clf_scaler = scaler if numeric_features else None
            st.session_state.clf_numeric_features = numeric_features
            st.session_state.clf_categorical_features = categorical_features
            st.session_state.clf_y_test = y_test
            st.session_state.clf_y_pred = y_pred
            st.session_state.clf_y_pred_proba = y_pred_proba
            st.session_state.clf_accuracy = accuracy
            st.session_state.clf_precision = precision
            st.session_state.clf_recall = recall
            st.session_state.clf_f1 = f1
            st.session_state.clf_cv_scores = cv_scores

            st.success("🎉 Model training completed!")

        except Exception as e:
            st.error(f"❌ Error in model training: {str(e)}")
            st.error(f"Details: {traceback.format_exc()}")
            return
        finally:
            progress_bar.empty()
    
    # Show tabs if model is trained
    if st.session_state.trained_classifier is not None:
        st.markdown("---")
        
        # Create tabs for results and prediction - ADDED BATCH PREDICTION TAB
        tab1, tab2, tab3 = st.tabs(["📊 Model Results & Analysis", "🔮 Single Prediction", "📦 Batch Prediction"])
        
        with tab1:
            st.subheader("📈 Model Performance")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{st.session_state.clf_accuracy:.3f}")
            col2.metric("Precision", f"{st.session_state.clf_precision:.3f}")
            col3.metric("Recall", f"{st.session_state.clf_recall:.3f}")
            col4.metric("F1-Score", f"{st.session_state.clf_f1:.3f}")
            st.info(f"📊 Cross-validated Accuracy: {np.mean(st.session_state.clf_cv_scores):.3f} (±{np.std(st.session_state.clf_cv_scores):.3f})")
            
            # Show confusion matrix and other plots
            plot_classification_results(
                st.session_state.clf_y_test, 
                st.session_state.clf_y_pred, 
                st.session_state.clf_y_pred_proba, 
                st.session_state.trained_classifier, 
                st.session_state.clf_features, 
                st.session_state.clf_label_encoders.get('target'), 
                st.session_state.clf_target_name
            )
            
            # Show model interpretation if available
            if hasattr(st.session_state.trained_classifier, 'coef_') or hasattr(st.session_state.trained_classifier, 'feature_importances_'):
                show_classification_interpretation(
                    st.session_state.trained_classifier, 
                    st.session_state.clf_features, 
                    st.session_state.clf_target_name, 
                    st.session_state.clf_label_encoders.get('target')
                )
        
        with tab2:
            st.subheader("🎯 Make Single Prediction")
            
            input_data = {}
            cols = st.columns(2)
            col_idx = 0
            
            for i, feature in enumerate(st.session_state.clf_features):
                with cols[col_idx]:
                    if feature in categorical_cols:
                        le = st.session_state.clf_label_encoders.get(feature)
                        if le:
                            selected = st.selectbox(f"Select {feature}", le.classes_, key=f"{feature}_input")
                            input_data[feature] = le.transform([selected])[0]
                    elif feature in numeric_cols:
                        # Get min, max, and median values for better input experience
                        min_val = float(st.session_state.clf_df_clean[feature].min())
                        max_val = float(st.session_state.clf_df_clean[feature].max())
                        median_val = float(st.session_state.clf_df_clean[feature].median())
                        
                        val = st.number_input(
                            f"Enter {feature}", 
                            value=median_val,
                            min_value=min_val,
                            max_value=max_val,
                            step=(max_val - min_val) / 100 if max_val > min_val else 0.1,
                            key=f"{feature}_input"
                        )
                        input_data[feature] = val
                col_idx = (col_idx + 1) % 2

            # Add some spacing
            st.markdown("---")
            
            # Prediction button with better styling
            if st.button("🚀 Predict Category", type="primary", use_container_width=True):
                try:
                    input_df = pd.DataFrame([input_data])
                    
                    # Apply scaling to numeric features if scaler exists
                    if st.session_state.clf_scaler and st.session_state.clf_numeric_features:
                        input_df[st.session_state.clf_numeric_features] = st.session_state.clf_scaler.transform(
                            input_df[st.session_state.clf_numeric_features])

                    model = st.session_state.trained_classifier
                    prediction = model.predict(input_df)[0]
                    predicted_class = st.session_state.clf_label_encoders['target'].inverse_transform([prediction])[0]
                    
                    # Show prediction result with better formatting
                    st.success(f"✅ **Predicted {st.session_state.clf_target_name}:** `{predicted_class}`")
                    
                    # Show prediction probabilities if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_df)[0]
                        target_encoder = st.session_state.clf_label_encoders['target']
                        
                        st.subheader("🎲 Prediction Probabilities")
                        prob_data = []
                        for i, prob in enumerate(proba):
                            class_name = target_encoder.inverse_transform([i])[0]
                            prob_data.append({'Class': class_name, 'Probability': prob})
                        
                        prob_df = pd.DataFrame(prob_data)
                        prob_df = prob_df.sort_values('Probability', ascending=False)
                        
                        # Display as a nice table with progress bars
                        for _, row in prob_df.iterrows():
                            col1, col2 = st.columns([1, 3])
                            col1.write(f"**{row['Class']}**")
                            col2.progress(row['Probability'], text=f"{row['Probability']:.1%}")
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        st.write("**Input Values Used:**")
                        for feature in st.session_state.clf_features:
                            if feature in categorical_cols:
                                le = st.session_state.clf_label_encoders.get(feature)
                                if le:
                                    selected_value = le.inverse_transform([input_data[feature]])[0]
                                    st.write(f"**{feature}:** {selected_value}")
                            else:
                                st.write(f"**{feature}:** {input_data[feature]}")
                except Exception as e:
                    st.error(f"❌ Error in making prediction: {str(e)}")

        with tab3:  # NEW BATCH PREDICTION TAB
            st.subheader("📊 Batch Prediction & Trend Analysis")
            
            # File upload section for test data
            st.markdown("### 1️⃣ Upload Test Dataset")
            test_file = st.file_uploader(
                "Upload CSV with input features (same structure as training data)",
                type=['csv'],
                key="batch_pred_file"
            )
            
            batch_df = None
            predictions_df = None
            actuals_df = None
            
            if test_file is not None:
                try:
                    batch_df = pd.read_csv(test_file)
                    st.success(f"✅ Test data loaded! {batch_df.shape[0]} rows")
                    
                    # Check if required features exist
                    missing_features = [f for f in st.session_state.clf_features if f not in batch_df.columns]
                    if missing_features:
                        st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                        batch_df = None
                    else:
                        # Preprocess the test data
                        test_processed = batch_df[st.session_state.clf_features].copy()
                        
                        # Apply the same preprocessing as training
                        for col in st.session_state.clf_categorical_features:
                            le = st.session_state.clf_label_encoders.get(col)
                            if le:
                                # Handle unseen categories
                                test_processed[col] = test_processed[col].apply(
                                    lambda x: x if x in le.classes_ else 'Unknown'
                                )
                                test_processed[col] = le.transform(test_processed[col])
                        
                        if st.session_state.clf_scaler and st.session_state.clf_numeric_features:
                            test_processed[st.session_state.clf_numeric_features] = (
                                st.session_state.clf_scaler.transform(
                                    test_processed[st.session_state.clf_numeric_features]
                                )
                            )
                        
                        # Make predictions
                        model = st.session_state.trained_classifier
                        predictions = model.predict(test_processed)
                        probabilities = model.predict_proba(test_processed) if hasattr(model, 'predict_proba') else None
                        
                        # Create predictions dataframe
                        predictions_df = batch_df.copy()
                        target_encoder = st.session_state.clf_label_encoders['target']
                        predictions_df['Predicted_Class'] = target_encoder.inverse_transform(predictions)
                        
                        # Add probabilities for each class
                        if probabilities is not None:
                            for i, class_name in enumerate(target_encoder.classes_):
                                predictions_df[f'Probability_{class_name}'] = probabilities[:, i]
                        
                        st.session_state.batch_predictions = predictions_df
                        
                        # Show prediction distribution
                        st.subheader("📈 Prediction Distribution")
                        class_counts = predictions_df['Predicted_Class'].value_counts().reset_index()
                        class_counts.columns = ['Class', 'Count']
                        
                        fig = px.bar(
                            class_counts, 
                            x='Class', 
                            y='Count', 
                            color='Class',
                            title="Predicted Class Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show sample predictions
                        st.subheader("🧪 Sample Predictions")
                        st.dataframe(predictions_df.head(5), use_container_width=True)
                        
                        # Download predictions
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Add actuals upload section
                        st.markdown("---")
                        st.subheader("🔍 Compare with Actual Values")
                        actuals_file = st.file_uploader(
                            "Upload CSV with actual target values (same row order)",
                            type=['csv'],
                            key="actuals_file"
                        )
                        
                        if actuals_file is not None:
                            actuals_df =pd.read_csv(actuals_file)
                            st.success(f"✅ Actual values loaded! {actuals_df.shape[0]} rows")
                            
                            # Check if actuals match the predictions
                            if actuals_df.shape[0] != predictions_df.shape[0]:
                                st.error("❌ The number of rows in actuals does not match predictions.")
                            else:
                                # Merge predictions with actuals for comparison
                                comparison_df = predictions_df.copy()
                                comparison_df['Actual_Class'] = actuals_df.iloc[:, 0]  # Assuming the first column contains actual values
                                
                                # Calculate accuracy
                                accuracy = accuracy_score(comparison_df['Actual_Class'], comparison_df['Predicted_Class'])
                                st.metric("Accuracy of Predictions", f"{accuracy:.3f}")
                                
                                # Show comparison table
                                st.subheader("📊 Predictions vs Actuals")
                                st.dataframe(comparison_df[['Actual_Class', 'Predicted_Class']], use_container_width=True)
                                
                                # Show confusion matrix
                                cm = confusion_matrix(comparison_df['Actual_Class'], comparison_df['Predicted_Class'])
                                fig_cm = px.imshow(cm, 
                                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                                    x=target_encoder.classes_,
                                                    y=target_encoder.classes_,
                                                    color_continuous_scale='Blues',
                                                    title="Confusion Matrix")
                                st.plotly_chart(fig_cm, use_container_width=True)

                        else:
                            st.warning("⚠️ Please upload actual values to compare with predictions.")
                except Exception as e:
                    st.error(f"❌ Error loading test data: {str(e)}")
