import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import json
import io
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    st.warning("Boruta package not installed. Boruta algorithm will not be available.")


def feature_selection_module(df, numeric_cols, categorical_cols):
    st.header("⚙️ Feature Selection Methods")

    available_methods = [
        "Correlation Analysis",
        "Mutual Information",
        "Recursive Feature Elimination (RFE)",
        "Model-Based Selection",
        "Sequential Feature Selection"
    ]
    
    if BORUTA_AVAILABLE:
        available_methods.append("Boruta Algorithm")

    method = st.selectbox("Choose Method", available_methods)

    target = st.selectbox("🎯 Target Variable", numeric_cols)
    features = st.multiselect(
        "🧮 Input Features",
        [col for col in numeric_cols + categorical_cols if col != target],
        default=[col for col in numeric_cols + categorical_cols if col != target][:2]
    )

    if not features:
        st.warning("⚠️ Please select at least one input feature")
        return

    df_clean = df.dropna(subset=[target] + features).copy()
    if df_clean.empty:
        st.error("❌ No valid data remaining after cleaning missing values")
        return
        
    X = df_clean[features]
    y = df_clean[target]

    # Encode categorical features for feature selection methods that require numeric input
    X_enc = pd.get_dummies(X, drop_first=True)

    if method == "Correlation Analysis":
        corr = X_enc.corrwith(y)
        st.write("Correlation with target variable:")
        corr_df = pd.DataFrame({'Feature': corr.index, 'Correlation': corr.values})
        st.dataframe(corr_df)
        
        fig = px.bar(x=corr.index, y=corr.values, 
                     labels={'x': 'Features', 'y': 'Correlation'},
                     title="Feature Correlation with Target")
        st.plotly_chart(fig, use_container_width=True)

    elif method == "Mutual Information":
        try:
            mi = mutual_info_regression(X_enc, y, random_state=42)
            st.write("Mutual Information scores:")
            mi_df = pd.DataFrame({'Feature': X_enc.columns, 'MI_Score': mi})
            st.dataframe(mi_df)
            
            fig = px.bar(x=X_enc.columns, y=mi,
                         labels={'x': 'Features', 'y': 'MI Score'},
                         title="Mutual Information Scores")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error calculating Mutual Information: {str(e)}")

    elif method == "Recursive Feature Elimination (RFE)":
        try:
            model = LinearRegression()
            n_features = min(5, len(X_enc.columns))  # Select top 5 or all if less than 5
            rfe = RFE(model, n_features_to_select=n_features)
            rfe.fit(X_enc, y)
            selected_features = X_enc.columns[rfe.support_]
            rankings = rfe.ranking_
            
            st.write(f"Top {n_features} Features selected by RFE:")
            rfe_df = pd.DataFrame({
                'Feature': X_enc.columns,
                'Selected': rfe.support_,
                'Ranking': rankings
            })
            st.dataframe(rfe_df.sort_values('Ranking'))
        except Exception as e:
            st.error(f"Error in RFE: {str(e)}")

    elif method == "Model-Based Selection":
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_enc, y)
            importances = model.feature_importances_
            
            st.write("Feature Importances from Random Forest:")
            imp_df = pd.DataFrame({
                'Feature': X_enc.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            st.dataframe(imp_df)
            
            fig = px.bar(x=imp_df['Feature'], y=imp_df['Importance'],
                         labels={'x': 'Features', 'y': 'Importance'},
                         title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in model-based selection: {str(e)}")

    elif method == "Sequential Feature Selection":
        try:
            from sklearn.feature_selection import SequentialFeatureSelector
            model = LinearRegression()
            n_features = min(3, len(X_enc.columns))  # Select top 3 or all if less than 3
            sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward')
            sfs.fit(X_enc, y)
            selected_features = X_enc.columns[sfs.get_support()]
            
            st.write(f"Top {n_features} Features selected by Sequential Feature Selection:")
            sfs_df = pd.DataFrame({
                'Feature': X_enc.columns,
                'Selected': sfs.get_support()
            })
            st.dataframe(sfs_df[sfs_df['Selected']])
        except Exception as e:
            st.error(f"Error in Sequential Feature Selection: {str(e)}")

    elif method == "Boruta Algorithm" and BORUTA_AVAILABLE:
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            boruta_selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
            boruta_selector.fit(X_enc.values, y.values)
            
            selected_features = X_enc.columns[boruta_selector.support_]
            confirmed_features = X_enc.columns[boruta_selector.support_]
            tentative_features = X_enc.columns[boruta_selector.support_weak_]
            
            st.write("Boruta Algorithm Results:")
            boruta_df = pd.DataFrame({
                'Feature': X_enc.columns,
                'Confirmed': boruta_selector.support_,
                'Tentative': boruta_selector.support_weak_,
                'Ranking': boruta_selector.ranking_
            })
            st.dataframe(boruta_df.sort_values('Ranking'))
            
            st.write(f"**Confirmed Features ({len(confirmed_features)}):** {list(confirmed_features)}")
            if len(tentative_features) > 0:
                st.write(f"**Tentative Features ({len(tentative_features)}):** {list(tentative_features)}")
        except Exception as e:
            st.error(f"Error in Boruta Algorithm: {str(e)}")


def prediction_engine(df):
    st.title("🔮 Advanced Predictive Analytics")
    st.markdown("""
    *Predict future outcomes* using various regression models. 
    Features include model comparison, residual analysis, and prediction explanations.
    """)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("⚠️ Insufficient numeric columns for analysis. Need at least 2 numeric columns.")
        return

    with st.expander("📊 Data Overview", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Total Samples", df.shape[0])
        cols[1].metric("Numeric Features", len(numeric_cols))
        cols[2].metric("Categorical Features", len(categorical_cols))

    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_type = st.selectbox("Algorithm", [
            "Linear Regression",
            "Random Forest",
            "Polynomial Regression",
            "Ridge Regression",
            "Lasso Regression"
        ])
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        random_state = st.number_input("Random State", 0, 1000, 42)
        st.markdown("---")

        st.subheader("🎛️ Model Parameters")
        model_params = {}
        if model_type == "Random Forest":
            model_params['rf_trees'] = st.number_input("Number of Trees", 10, 500, 100, step=10, key="rf")
        elif model_type == "Polynomial Regression":
            model_params['poly_deg'] = st.slider("Polynomial Degree", 2, 6, 2, key="poly")
        elif model_type == "Ridge Regression":
            model_params['ridge_alpha'] = st.slider("Alpha (λ)", 0.01, 10.0, 1.0, 0.01, key="ridge")
        elif model_type == "Lasso Regression":
            model_params['lasso_alpha'] = st.slider("Alpha (λ)", 0.001, 1.0, 0.1, 0.001, key="lasso")

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("🎯 Target Variable", numeric_cols)
    with col2:
        features = st.multiselect(
            "🧮 Input Features",
            [col for col in numeric_cols + categorical_cols if col != target],
            default=[col for col in numeric_cols + categorical_cols if col != target][:2]
        )

    if not features:
        st.warning("⚠️ Please select at least one input feature")
        return

    df_clean = df.dropna(subset=[target] + features).copy()
    if df_clean.empty:
        st.error("❌ No valid data remaining after cleaning missing values")
        return

    X = df_clean[features]
    y = df_clean[target]

    # Convert categorical variables to dummy variables
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        X = pd.get_dummies(X, drop_first=True)
        st.info(f"🔠 Created dummy variables for categorical features: {categorical_features}")

    # Initialize session state
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
        st.session_state.model_features = None
        st.session_state.target_name = None
        st.session_state.model_type = None
        st.session_state.feature_names = None

    if st.button("🚀 Train & Evaluate Model"):
        progress_bar = st.progress(0)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size / 100,
                random_state=random_state
            )
            progress_bar.progress(20)

            model = create_model(model_type, model_params)
            progress_bar.progress(40)

            with st.spinner("🏋️ Training model..."):
                model.fit(X_train, y_train)
                progress_bar.progress(60)

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            progress_bar.progress(80)

            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            progress_bar.progress(100)

            # Store in session state
            st.session_state.trained_model = model
            st.session_state.model_features = features
            st.session_state.target_name = target
            st.session_state.model_type = model_type
            st.session_state.df_clean = df_clean
            st.session_state.feature_names = X.columns.tolist()

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.3f}")
            col2.metric("RMSE", f"{rmse:.3f}")
            col3.metric("R² Score", f"{r2:.3f}")
            col4.metric("CV R² (±std)", f"{np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

            plot_results(y_test, y_pred, model, X.columns.tolist())

            if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
                show_coefficients(model, X.columns.tolist())

            export_model_params(model, model_type, X.columns.tolist(), target)

        except Exception as e:
            st.error(f"❌ Error in model training: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")

    # Prediction interface
    if st.session_state.trained_model is not None:
        prediction_interface(
            st.session_state.trained_model,
            st.session_state.model_features,
            st.session_state.target_name,
            st.session_state.df_clean,
            st.session_state.feature_names
        )


def create_model(model_type, params):
    """Create and return the specified model with given parameters"""
    if model_type == "Linear Regression":
        return LinearRegression()
    elif model_type == "Random Forest":
        return RandomForestRegressor(n_estimators=params.get('rf_trees', 100), random_state=42)
    elif model_type == "Polynomial Regression":
        return make_pipeline(PolynomialFeatures(params.get('poly_deg', 2)), LinearRegression())
    elif model_type == "Ridge Regression":
        return Ridge(alpha=params.get('ridge_alpha', 1.0))
    elif model_type == "Lasso Regression":
        return Lasso(alpha=params.get('lasso_alpha', 0.1))
    return LinearRegression()


def plot_results(y_test, y_pred, model, features):
    """Create visualization tabs for model results"""
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance"])

    with tab1:
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Actual vs Predicted Values"
        )
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_shape(
            type="line", 
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        residuals = y_test - y_pred
        fig = px.scatter(
            x=y_pred, y=residuals,
            labels={'x': 'Predicted', 'y': 'Residuals'},
            title="Residual Analysis"
        )
        fig.add_hline(y=0, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            fig = px.bar(
                x=features, y=importance,
                labels={'x': 'Features', 'y': 'Importance'},
                title="Feature Importance"
            )
            fig.update_xaxes(tickangle=45)  # Fixed: update_xaxes instead of update_xaxis
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'coef_'):
            coefs = model.coef_ if hasattr(model, 'coef_') else model.named_steps['linearregression'].coef_
            fig = px.bar(
                x=features, y=coefs,
                labels={'x': 'Features', 'y': 'Coefficient'},
                title="Model Coefficients"
            )
            fig.update_xaxes(tickangle=45)  # Fixed: update_xaxes instead of update_xaxis
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance/coefficients not available for this model type")
            
def prediction_interface(model, original_features, target, df, model_features):
    """Interface for making single and batch predictions"""
    with st.expander("🔮 Make Predictions", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Single Prediction")
            inputs = {}
            
            # Create inputs for original features
            for feature in original_features:
                if feature in df.columns:
                    if df[feature].dtype in ['object', 'category']:
                        unique_vals = df[feature].dropna().unique()
                        inputs[feature] = st.selectbox(
                            f"{feature}",
                            options=unique_vals,
                            key=f"single_pred_{feature}"
                        )
                    else:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        median_val = float(df[feature].median())
                        inputs[feature] = st.number_input(
                            label=f"{feature}",
                            value=median_val,
                            min_value=min_val,
                            max_value=max_val,
                            step=(max_val - min_val) / 100 if max_val > min_val else 0.1,
                            key=f"single_pred_{feature}"
                        )

            if st.button("Predict Single", key="predict_single_button"):
                try:
                    input_df = pd.DataFrame([inputs])
                    
                    # Apply same preprocessing as training data
                    categorical_features = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if categorical_features:
                        input_df = pd.get_dummies(input_df, drop_first=True)
                    
                    # Ensure all model features are present
                    for col in model_features:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    
                    # Reorder columns to match model training
                    input_df = input_df[model_features]
                    
                    pred = model.predict(input_df)
                    st.success(f"Predicted {target}: **{pred[0]:.4f}**")
                    
                    st.write("**Input Values:**")
                    for feat, val in inputs.items():
                        st.write(f"- {feat}: {val}")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

        with col2:
            st.subheader("Batch Prediction")
            uploaded_file = st.file_uploader("Upload CSV file with features", type="csv", key="batch_file_uploader")
            if uploaded_file:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(batch_df.head(5))
                    
                    missing_cols = [col for col in original_features if col not in batch_df.columns]
                    if missing_cols:
                        st.error(f"Missing required feature columns in uploaded file: {missing_cols}")
                    else:
                        if st.button("Run Batch Prediction", key="run_batch_pred"):
                            try:
                                # Prepare batch data same way as training data
                                batch_features = batch_df[original_features].copy()
                                
                                # Apply preprocessing
                                categorical_features = batch_features.select_dtypes(include=['object', 'category']).columns.tolist()
                                if categorical_features:
                                    batch_features = pd.get_dummies(batch_features, drop_first=True)
                                
                                # Ensure all model features are present
                                for col in model_features:
                                    if col not in batch_features.columns:
                                        batch_features[col] = 0
                                
                                # Reorder columns to match model training
                                batch_features = batch_features[model_features]
                                
                                pred_batch = model.predict(batch_features)
                                result_df = batch_df.copy()
                                result_df[f'Predicted_{target}'] = pred_batch
                                
                                st.write("**Prediction Results:**")
                                st.dataframe(result_df)
                                
                                csv = result_df.to_csv(index=False).encode()
                                st.download_button(
                                    label="📥 Download Predictions CSV",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv",
                                    key="download_batch_pred"
                                )
                            except Exception as e:
                                st.error(f"Error during batch prediction: {str(e)}")
                                st.error("Please ensure all required features are present and in the correct format.")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")


def export_model_params(model, model_type, features, target_name):
    """Export model parameters as JSON"""
    model_info = {
        'model_type': model_type, 
        'features': features,
        'target': target_name
    }
    
    try:
        if model_type == "Random Forest":
            model_info['n_estimators'] = getattr(model, 'n_estimators', None)
            model_info['feature_importances'] = {}
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(features, model.feature_importances_):
                    model_info['feature_importances'][feat] = float(imp)
                    
        elif model_type == "Polynomial Regression":
            if hasattr(model, 'named_steps'):
                poly = model.named_steps.get('polynomialfeatures', None)
                linreg = model.named_steps.get('linearregression', None)
                if poly:
                    model_info['polynomial_degree'] = poly.degree
                if linreg and hasattr(linreg, 'intercept_'):
                    model_info['intercept'] = float(linreg.intercept_)
                    model_info['coefficients'] = [float(c) for c in linreg.coef_]
                    
        else:
            if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                model_info['intercept'] = float(model.intercept_)
                model_info['coefficients'] = {}
                for i, feat in enumerate(features):
                    if i < len(model.coef_):
                        model_info['coefficients'][feat] = float(model.coef_[i])
                
                # Create prediction formula for linear models
                if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
                    formula = f"{target_name} = {model.intercept_:.4f}"
                    for i, feat in enumerate(features):
                        if i < len(model.coef_):
                            coef = model.coef_[i]
                            sign = "+" if coef >= 0 else ""
                            formula += f" {sign} {coef:.4f}*{feat}"
                    model_info['prediction_formula'] = formula
                    
    except Exception as e:
        st.warning(f"Error extracting some model parameters: {str(e)}")
    
    json_str = json.dumps(model_info, indent=4)
    st.download_button(
        label="💾 Export Model Parameters (JSON)",
        data=json_str,
        file_name="model_parameters.json",
        mime="application/json",
        help="Download model parameters and info as JSON file"
    )


def show_coefficients(model, features):
    """Display model coefficients and interpretation"""
    with st.expander("📝 Model Interpretation"):
        target_name = st.session_state.get('target_name', 'Target')
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Random Forest feature importance
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                st.write("**Feature Importance:**")
                st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}))
                
            elif hasattr(model, 'coef_'):
                # Linear model coefficients
                if hasattr(model, 'named_steps') and 'linearregression' in model.named_steps:
                    # Polynomial regression
                    coefs = model.named_steps['linearregression'].coef_
                    intercept = model.named_steps['linearregression'].intercept_
                    st.info("⚠️ This is a polynomial model. Coefficients represent polynomial terms.")
                    
                    coef_df = pd.DataFrame({
                        'Term': [f'Term_{i}' for i in range(len(coefs))],
                        'Coefficient': coefs
                    })
                    st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}))
                    
                else:
                    # Regular linear model
                    coefficients = model.coef_
                    intercept = model.intercept_
                    
                    coef_df = pd.DataFrame({
                        'Feature': features[:len(coefficients)],
                        'Coefficient': coefficients[:len(features)]
                    })
                    st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}))
                    
                    # Show prediction formula
                    formula = f"**Prediction Formula:** {target_name} = {intercept:.4f}"
                    for i, feat in enumerate(features[:len(coefficients)]):
                        coef = coefficients[i]
                        sign = " + " if coef >= 0 else " - "
                        formula += f"{sign}{abs(coef):.4f} × {feat}"
                    st.markdown(formula)
                    
        except Exception as e:
            st.error(f"Error displaying coefficients: {str(e)}")


def main():
    st.set_page_config(page_title="Regression Analytics Dashboard", layout="wide")
    st.title("📈 Regression Analytics Dashboard")
    st.markdown("Upload your data and explore advanced regression analytics with feature selection and predictive modeling.")

    uploaded_file = st.file_uploader("Upload CSV file for analysis", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")

            # Show basic info about the dataset
            with st.expander("📋 Dataset Info", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**First 5 rows:**")
                    st.dataframe(df.head())
                with col2:
                    st.write("**Data types:**")
                    st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Type', 'index': 'Column'}))

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if not numeric_cols:
                st.error("❌ No numeric columns found in the dataset. Please upload a dataset with numeric columns.")
                return
            elif len(numeric_cols) < 2:
                st.error("❌ Need at least 2 numeric columns for regression analysis.")
                return
            else:
                # Feature Selection Module
                feature_selection_module(df, numeric_cols, categorical_cols)
                
                st.markdown("---")
                
                # Prediction Engine
                prediction_engine(df)

        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {str(e)}")
            st.error("Please ensure the file is a valid CSV format.")
    else:
        st.info("👆 Please upload a CSV file to begin analysis.")
        
        # Show example of expected data format
        with st.expander("💡 Expected Data Format", expanded=False):
            st.markdown("""
            Your CSV file should contain:
            - **Numeric columns**: For regression analysis (e.g., price, age, income)
            - **Target variable**: The variable you want to predict
            - **Feature variables**: Variables used to make predictions
            
            Example structure:
            ```
            feature1,feature2,category,target
            1.2,3.4,A,10.5
            2.1,4.3,B,12.3
            ...
            ```
            """)


if __name__ == "__main__":
    main()