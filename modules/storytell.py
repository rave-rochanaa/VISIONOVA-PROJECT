import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def detect_seasonality(df, date_col, value_col):
    """Detect monthly/quarterly patterns in time series data"""
    try:
        df = df.copy()
        df['month'] = df[date_col].dt.month
        monthly = df.groupby('month')[value_col].mean()
        
        # Calculate seasonality strength
        seasonal_strength = monthly.std() / monthly.mean()
        
        if seasonal_strength > 0.2:
            peak_month = monthly.idxmax()
            low_month = monthly.idxmin()
            return {
                'strength': seasonal_strength,
                'peak_month': peak_month,
                'low_month': low_month,
                'peak_value': monthly.max(),
                'low_value': monthly.min()
            }
        return None
    except Exception as e:
        print(f"Seasonality detection error: {e}")
        return None

def detect_trends(df, date_col, value_col):
    """Detect upward/downward trends in time series data"""
    try:
        df = df.copy().sort_values(date_col)
        x = np.arange(len(df))
        y = df[value_col].values
        
        # Calculate linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Calculate trend strength
        trend_strength = abs(slope) / (y.std() if y.std() > 0 else 1)
        
        return {
            'slope': slope,
            'strength': trend_strength,
            'direction': 'upward' if slope > 0 else 'downward'
        }
    except Exception as e:
        print(f"Trend detection error: {e}")
        return None

def detect_correlations(df):
    """Detect strong correlations between numerical columns"""
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            return []
        
        corr_matrix = df[num_cols].corr().abs()
        correlations = []
        
        # Find significant correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if corr_value > 0.7:
                    correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value,
                        'type': 'strong positive'
                    })
                elif corr_value < -0.7:
                    correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value,
                        'type': 'strong negative'
                    })
        
        return correlations
    except Exception as e:
        print(f"Correlation detection error: {e}")
        return []

def detect_outliers(df):
    """Detect outliers in numerical columns using IQR method"""
    try:
        num_cols = df.select_dtypes(include=np.number).columns
        outliers = []
        
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outlier_count > 0:
                outliers.append({
                    'feature': col,
                    'count': outlier_count,
                    'percentage': outlier_count / len(df)
                })
                
        
        return outliers
    except Exception as e:
        print(f"Outlier detection error: {e}")
        return []

def detect_missing_values(df):
    """Detect missing values in all columns"""
    try:
        missing = df.isnull().sum()
        missing_pct = missing / len(df)
        return [
            {'feature': col, 'count': missing[col], 'percentage': pct}
            for col, pct in missing_pct.items() if pct > 0
        ]
    except Exception as e:
        print(f"Missing value detection error: {e}")
        return []

def detect_categorical_distributions(df):
    """Detect distributions in categorical columns"""
    try:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        distributions = []
        
        for col in cat_cols:
            counts = df[col].value_counts().nlargest(5)
            if len(counts) > 0:
                distributions.append({
                    'feature': col,
                    'top_values': counts.to_dict()
                })
        
        return distributions
    except Exception as e:
        print(f"Categorical distribution error: {e}")
        return []

def generate_data_story(df):
    """Generate a comprehensive data story with visualizations"""
    if df is None or df.empty:
        return "No data available to generate story."
    
    story = []
    
    # 1. Dataset Overview
    story.append({
        'section': "📚 Data Overview",
        'content': f"The dataset contains *{df.shape[0]:,} records* and *{df.shape[1]} features*.",
        'visualization': None
    })
    
    # Detect date columns for time-based analysis
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # 2. Time Range Analysis (if date columns exist)
    if date_cols:
        date_col = date_cols[0]  # Use first date column
        min_date = df[date_col].min().strftime('%Y-%m-%d')
        max_date = df[date_col].max().strftime('%Y-%m-%d')
        time_span = (df[date_col].max() - df[date_col].min()).days
        
        story.append({
            'section': "📅 Time Coverage",
            'content': f"Data spans from *{min_date}* to *{max_date}* ({time_span} days).",
            'visualization': None
        })
    
    # 3. Missing Values Analysis
    missing_values = detect_missing_values(df)
    if missing_values:
        missing_content = []
        for item in missing_values:
            missing_content.append(
                f"- *{item['feature']}*: {item['count']} missing values "
                f"({item['percentage']:.1%})"
            )
        
        # Create missing values visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        missing_data = pd.DataFrame(missing_values).set_index('feature')
        missing_data['percentage'].plot(kind='bar', ax=ax, color='salmon')
        plt.title('Missing Values by Feature')
        plt.ylabel('Percentage Missing')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        story.append({
            'section': "❓ Missing Values",
            'content': "Several features contain missing data:\n" + "\n".join(missing_content),
            'visualization': fig
        })
    else:
        story.append({
            'section': "✅ Data Completeness",
            'content': "No missing values detected in any features.",
            'visualization': None
        })
    
    # 4. Numerical Feature Analysis
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        num_content = []
        num_insights = []
        
        # Basic statistics
        stats_df = df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
        stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Median', 'Max']
        
        for col in num_cols:
            stats = stats_df.loc[col]
            insight = f"- *{col}*: Mean={stats['Mean']:.2f}, Std Dev={stats['Std Dev']:.2f}, " \
                      f"Range=[{stats['Min']:.2f} to {stats['Max']:.2f}]"
            num_content.append(insight)
        
        # Add distribution visualization
        fig, axes = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(10, 3*len(num_cols)))
        if len(num_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(num_cols):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        
        story.append({
            'section': "🔢 Numerical Features",
            'content': "Key statistics for numerical features:\n" + "\n".join(num_content),
            'visualization': fig
        })
        
        # 5. Correlation Analysis
        correlations = detect_correlations(df)
        if correlations:
            corr_content = []
            for corr in correlations:
                corr_content.append(
                    f"- *{corr['feature1']}* and *{corr['feature2']}* have a "
                    f"{corr['type']} relationship (r={corr['correlation']:.2f})"
                )
            
            # Create correlation heatmap
            corr_matrix = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                        vmin=-1, vmax=1, ax=ax)
            plt.title('Feature Correlation Matrix')
            
            story.append({
                'section': "🔗 Feature Relationships",
                'content': "Strong correlations detected between features:\n" + "\n".join(corr_content),
                'visualization': fig
            })
        
        # 6. Trend and Seasonality Analysis
        if date_cols:
            date_col = date_cols[0]
            trend_season_content = []
            
            for num_col in num_cols[:3]:  # Limit to first 3 numerical columns
                # Detect trends
                trend = detect_trends(df, date_col, num_col)
                if trend and trend['strength'] > 0.1:
                    trend_season_content.append(
                        f"- *{num_col}* shows a *{trend['direction']} trend* "
                        f"(strength: {trend['strength']:.2f})"
                    )
                
                # Detect seasonality
                season = detect_seasonality(df, date_col, num_col)
                if season:
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    trend_season_content.append(
                        f"- *{num_col}* shows strong seasonality with peak in "
                        f"{month_names[season['peak_month']-1]} "
                        f"({season['peak_value']:.2f}) and low in "
                        f"{month_names[season['low_month']-1]} "
                        f"({season['low_value']:.2f})"
                    )
            
            if trend_season_content:
                # Create time series plot
                fig, ax = plt.subplots(figsize=(12, 6))
                for num_col in num_cols[:3]:
                    sns.lineplot(data=df, x=date_col, y=num_col, ax=ax, label=num_col)
                plt.title('Time Series Trends')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                story.append({
                    'section': "📈 Trends & Seasonality",
                    'content': "Time-based patterns detected:\n" + "\n".join(trend_season_content),
                    'visualization': fig
                })
        
        # 7. Outlier Detection
        outliers = detect_outliers(df)
        if outliers:
            outlier_content = []
            for outlier in outliers:
                outlier_content.append(
                    f"- *{outlier['feature']}*: {outlier['count']} outliers "
                    f"({outlier['percentage']:.1%} of data)"
                )
            
            # Create boxplot visualization
            fig, axes = plt.subplots(nrows=len(outliers), ncols=1, figsize=(10, 3*len(outliers)))
            if len(outliers) == 1:
                axes = [axes]
            
            for i, outlier in enumerate(outliers[:3]):  # Limit to first 3
                sns.boxplot(x=df[outlier['feature']], ax=axes[i])
                axes[i].set_title(f'Outliers in {outlier["feature"]}')
            
            plt.tight_layout()
            
            story.append({
                'section': "⚠️ Potential Outliers",
                'content': "Possible outliers detected in:\n" + "\n".join(outlier_content),
                'visualization': fig
            })
    
    # 8. Categorical Analysis
    cat_distributions = detect_categorical_distributions(df)
    if cat_distributions:
        cat_content = []
        
        for dist in cat_distributions:
            top_values = list(dist['top_values'].items())[:3]
            value_str = ", ".join([f"{k} ({v})" for k, v in top_values])
            cat_content.append(f"- *{dist['feature']}*: {value_str}")
        
        # Create categorical distribution visualization
        fig, axes = plt.subplots(nrows=len(cat_distributions), ncols=1, 
                                figsize=(10, 4*len(cat_distributions)))
        if len(cat_distributions) == 1:
            axes = [axes]
        
        for i, dist in enumerate(cat_distributions[:3]):  # Limit to first 3
            top_values = pd.Series(dist['top_values'])
            top_values.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Top Values in {dist["feature"]}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        story.append({
            'section': "🏷 Categorical Distributions",
            'content': "Top values in categorical features:\n" + "\n".join(cat_content),
            'visualization': fig
        })
    
    # 9. Key Takeaways
    key_takeaways = ["### 💎 Key Takeaways"]
    
    if len(num_cols) >= 3:
        key_takeaways.append("- Dataset contains multiple numerical dimensions for analysis")
    
    if cat_distributions:
        key_takeaways.append("- Categorical variables provide segmentation opportunities")
    
    if date_cols:
        key_takeaways.append("- Time-based features enable trend and seasonality analysis")
    
    if not missing_values:
        key_takeaways.append("- Complete dataset with no missing values")
    elif any(mv['percentage'] > 0.1 for mv in missing_values):
        key_takeaways.append("- Significant missing values present - consider imputation")
    
    if outliers:
        key_takeaways.append("- Potential outliers detected that may need investigation")
    
    if correlations:
        key_takeaways.append("- Strong correlations between features suggest relationships worth exploring")
    
    story.append({
        'section': "💎 Key Takeaways",
        'content': "\n".join(key_takeaways),
        'visualization': None
    })
    
    return story