import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional
import re
from datetime import datetime
import json

# Configure Streamlit page (must be first command)


class AnalyticsChatbot:
    """Advanced analytics chatbot with natural language processing"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.query_patterns = self._load_query_patterns()
        self.context_memory = []
        
    def _load_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Define response templates for different scenarios"""
        return {
            "greeting": {
                "template": "Hello! I'm your analytics assistant. I can help you analyze your data. What would you like to explore?",
                "suggestions": ["Show me sales trends", "Compare categories", "Find top performers", "Analyze by region"]
            },
            "data_overview": {
                "template": "I found {rows} rows and {cols} columns in your dataset. Key metrics include: {metrics}. Categories: {categories}.",
                "chart_type": "summary"
            },
            "comparison": {
                "template": "Comparing {metric} across {dimension}: The top performer is {top_item} with {top_value:,.2f}, followed by {second_item} with {second_value:,.2f}.",
                "chart_type": "bar"
            },
            "trend_analysis": {
                "template": "Analyzing {metric} trends over {time_dim}: {trend_description}. The {direction} trend shows {change_percent:.1f}% change.",
                "chart_type": "line"
            },
            "top_bottom": {
                "template": "Top {n} {dimension} by {metric}:\n{results}",
                "chart_type": "bar"
            },
            "correlation": {
                "template": "Correlation analysis between {var1} and {var2}: {correlation:.3f} ({strength}). {interpretation}",
                "chart_type": "scatter"
            },
            "summary_stats": {
                "template": "{metric} summary: Total: {total:,.2f}, Average: {avg:.2f}, Median: {median:.2f}, Std Dev: {std:.2f}",
                "chart_type": "box"
            },
            "filter_results": {
                "template": "Filtered results for {filter_desc}: Found {count} records. {additional_info}",
                "chart_type": "filtered_view"
            },
            "error": {
                "template": "I couldn't process that query. Try asking about: {suggestions}",
                "suggestions": ["totals", "averages", "trends", "comparisons", "top performers"]
            },
            "no_data": {
                "template": "I don't see any data matching those criteria. Would you like to try a different filter or metric?",
                "suggestions": ["Remove filters", "Try different metric", "Check data range"]
            }
        }
    
    def _load_query_patterns(self) -> Dict[str, Dict]:
        """Define regex patterns for query understanding"""
        return {
            "aggregation": {
                "sum": r"\b(total|sum|add up|combined?)\b",
                "mean": r"\b(average|avg|mean)\b",
                "count": r"\b(count|number of|how many)\b",
                "max": r"\b(max|maximum|highest|best|top|largest)\b",
                "min": r"\b(min|minimum|lowest|worst|bottom|smallest)\b",
                "median": r"\b(median|middle)\b"
            },
            "time_keywords": r"\b(trend|over time|by month|by year|by day|time series|temporal)\b",
            "comparison": r"\b(compare|vs|versus|against|between|difference)\b",
            "correlation": r"\b(correlat|relationship|association|connect)\b",
            "filter": r"\b(where|filter|only|exclude|include|for)\b",
            "top_bottom": r"\b(top \d+|bottom \d+|best \d+|worst \d+|\d+ highest|\d+ lowest)\b",
            "percentage": r"\b(percent|percentage|proportion|ratio)\b",
            "distribution": r"\b(distribution|spread|range|variance)\b"
        }
    
    def understand_query(self, query: str, df: pd.DataFrame, available_metrics: List[str], 
                        available_dimensions: List[str]) -> Dict[str, Any]:
        """Enhanced query understanding with context awareness"""
        query_lower = query.lower().strip()
        
        # Initialize query understanding
        understanding = {
            "intent": "unknown",
            "aggregation": "sum",
            "metrics": [],
            "dimensions": [],
            "filters": {},
            "limit": None,
            "time_analysis": False,
            "comparison": False,
            "correlation": False,
            "confidence": 0.0
        }
        
        # Detect aggregation function
        for agg_func, pattern in self.query_patterns["aggregation"].items():
            if re.search(pattern, query_lower):
                understanding["aggregation"] = agg_func
                understanding["confidence"] += 0.2
                break
        
        # Detect metrics (prioritize exact matches)
        for metric in available_metrics:
            if metric.lower() in query_lower:
                understanding["metrics"].append(metric)
                understanding["confidence"] += 0.3
        
        # If no exact metric match, try partial matches
        if not understanding["metrics"]:
            for metric in available_metrics:
                metric_words = metric.lower().split()
                if any(word in query_lower for word in metric_words):
                    understanding["metrics"].append(metric)
                    understanding["confidence"] += 0.1
        
        # Detect dimensions
        for dimension in available_dimensions:
            if dimension.lower() in query_lower:
                understanding["dimensions"].append(dimension)
                understanding["confidence"] += 0.2
        
        # Detect intent patterns
        if re.search(self.query_patterns["time_keywords"], query_lower):
            understanding["time_analysis"] = True
            understanding["intent"] = "trend_analysis"
            understanding["confidence"] += 0.3
        
        if re.search(self.query_patterns["comparison"], query_lower):
            understanding["comparison"] = True
            understanding["intent"] = "comparison"
            understanding["confidence"] += 0.3
        
        if re.search(self.query_patterns["correlation"], query_lower):
            understanding["correlation"] = True
            understanding["intent"] = "correlation"
            understanding["confidence"] += 0.3
        
        # Detect top/bottom queries
        top_bottom_match = re.search(self.query_patterns["top_bottom"], query_lower)
        if top_bottom_match:
            understanding["intent"] = "top_bottom"
            understanding["limit"] = int(re.search(r'\d+', top_bottom_match.group()).group())
            understanding["confidence"] += 0.4
        
        # Detect filters
        if "where" in query_lower or "filter" in query_lower:
            understanding["intent"] = "filter_results"
            understanding["confidence"] += 0.2
        
        # Default fallbacks
        if not understanding["metrics"] and available_metrics:
            understanding["metrics"] = [available_metrics[0]]
        
        if not understanding["dimensions"] and available_dimensions:
            understanding["dimensions"] = [available_dimensions[0]]
        
        # Determine final intent if not set
        if understanding["intent"] == "unknown":
            if understanding["metrics"] and understanding["dimensions"]:
                understanding["intent"] = "comparison"
            elif understanding["metrics"]:
                understanding["intent"] = "summary_stats"
        
        return understanding
    
    def execute_query(self, understanding: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the understood query and return results"""
        try:
            intent = understanding["intent"]
            metrics = understanding["metrics"]
            dimensions = understanding["dimensions"]
            agg_func = understanding["aggregation"]
            
            if intent == "comparison" and metrics and dimensions:
                return self._execute_comparison(df, metrics[0], dimensions[0], agg_func, understanding.get("limit", 10))
            
            elif intent == "trend_analysis" and metrics:
                time_dim = self._find_time_dimension(df, dimensions)
                if time_dim:
                    return self._execute_trend_analysis(df, metrics[0], time_dim, agg_func)
                else:
                    return self._execute_comparison(df, metrics[0], dimensions[0] if dimensions else None, agg_func)
            
            elif intent == "top_bottom" and metrics and dimensions:
                return self._execute_top_bottom(df, metrics[0], dimensions[0], agg_func, understanding["limit"])
            
            elif intent == "correlation" and len(metrics) >= 2:
                return self._execute_correlation(df, metrics[0], metrics[1])
            
            elif intent == "summary_stats" and metrics:
                return self._execute_summary_stats(df, metrics[0], dimensions[0] if dimensions else None)
            
            else:
                return self._execute_data_overview(df)
                
        except Exception as e:
            return {"type": "error", "error": str(e), "data": None}
    
    def _execute_comparison(self, df: pd.DataFrame, metric: str, dimension: str, agg_func: str, limit: int = 10) -> Dict:
        """Execute comparison analysis"""
        if dimension is None:
            # Overall metric
            result = df[metric].agg(agg_func)
            return {
                "type": "summary_stats",
                "metric": metric,
                "result": result,
                "data": pd.DataFrame({metric: [result]}),
                "message": f"Overall {agg_func} of {metric}: {result:,.2f}"
            }
        
        grouped = df.groupby(dimension)[metric].agg(agg_func).sort_values(ascending=False).head(limit)
        
        return {
            "type": "comparison",
            "metric": metric,
            "dimension": dimension,
            "agg_func": agg_func,
            "data": grouped,
            "top_item": grouped.index[0],
            "top_value": grouped.iloc[0],
            "second_item": grouped.index[1] if len(grouped) > 1 else "N/A",
            "second_value": grouped.iloc[1] if len(grouped) > 1 else 0
        }
    
    def _execute_trend_analysis(self, df: pd.DataFrame, metric: str, time_dim: str, agg_func: str) -> Dict:
        """Execute trend analysis"""
        grouped = df.groupby(time_dim)[metric].agg(agg_func).sort_index()
        
        # Calculate trend
        if len(grouped) > 1:
            change_percent = ((grouped.iloc[-1] - grouped.iloc[0]) / grouped.iloc[0]) * 100
            direction = "upward" if change_percent > 0 else "downward"
            trend_desc = f"showing {direction} movement"
        else:
            change_percent = 0
            direction = "stable"
            trend_desc = "insufficient data for trend analysis"
        
        return {
            "type": "trend_analysis",
            "metric": metric,
            "time_dim": time_dim,
            "data": grouped,
            "trend_description": trend_desc,
            "direction": direction,
            "change_percent": abs(change_percent)
        }
    
    def _execute_top_bottom(self, df: pd.DataFrame, metric: str, dimension: str, agg_func: str, limit: int) -> Dict:
        """Execute top/bottom analysis"""
        grouped = df.groupby(dimension)[metric].agg(agg_func).sort_values(ascending=False)
        top_results = grouped.head(limit)
        
        results_text = "\n".join([f"• {idx}: {value:,.2f}" for idx, value in top_results.items()])
        
        return {
            "type": "top_bottom",
            "metric": metric,
            "dimension": dimension,
            "n": limit,
            "data": top_results,
            "results": results_text
        }
    
    def _execute_correlation(self, df: pd.DataFrame, var1: str, var2: str) -> Dict:
        """Execute correlation analysis"""
        correlation = df[var1].corr(df[var2])
        
        if abs(correlation) > 0.7:
            strength = "strong"
            interpretation = "These variables are highly related."
        elif abs(correlation) > 0.3:
            strength = "moderate"
            interpretation = "These variables show some relationship."
        else:
            strength = "weak"
            interpretation = "These variables have little relationship."
        
        return {
            "type": "correlation",
            "var1": var1,
            "var2": var2,
            "correlation": correlation,
            "strength": strength,
            "interpretation": interpretation,
            "data": df[[var1, var2]]
        }
    
    def _execute_summary_stats(self, df: pd.DataFrame, metric: str, dimension: str = None) -> Dict:
        """Execute summary statistics"""
        if dimension:
            stats_data = df.groupby(dimension)[metric].agg(['sum', 'mean', 'median', 'std']).round(2)
            return {
                "type": "grouped_summary",
                "metric": metric,
                "dimension": dimension,
                "data": stats_data
            }
        else:
            return {
                "type": "summary_stats",
                "metric": metric,
                "total": df[metric].sum(),
                "avg": df[metric].mean(),
                "median": df[metric].median(),
                "std": df[metric].std(),
                "data": df[[metric]]
            }
    
    def _execute_data_overview(self, df: pd.DataFrame) -> Dict:
        """Execute data overview"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            "type": "data_overview",
            "rows": len(df),
            "cols": len(df.columns),
            "metrics": numeric_cols,
            "categories": categorical_cols,
            "data": df.head()
        }
    
    def _find_time_dimension(self, df: pd.DataFrame, dimensions: List[str]) -> Optional[str]:
        """Find time-related dimension"""
        time_keywords = ['date', 'time', 'month', 'year', 'day', 'week']
        
        for dim in dimensions:
            if any(keyword in dim.lower() for keyword in time_keywords):
                return dim
        
        # Check for datetime columns
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                return col
        
        return dimensions[0] if dimensions else None
    
    def generate_response(self, result: Dict[str, Any]) -> Tuple[str, Dict]:
        """Generate natural language response with visualization config"""
        result_type = result.get("type", "error")
        
        if result_type == "error":
            template = self.response_templates["error"]["template"]
            return template.format(suggestions=", ".join(self.response_templates["error"]["suggestions"])), {}
        
        elif result_type == "comparison":
            template = self.response_templates["comparison"]["template"]
            response = template.format(**result)
            viz_config = {
                "type": "bar",
                "data": result["data"],
                "title": f"{result['metric']} by {result['dimension']}"
            }
            return response, viz_config
        
        elif result_type == "trend_analysis":
            template = self.response_templates["trend_analysis"]["template"]
            response = template.format(**result)
            viz_config = {
                "type": "line",
                "data": result["data"],
                "title": f"{result['metric']} Trend Over {result['time_dim']}"
            }
            return response, viz_config
        
        elif result_type == "top_bottom":
            template = self.response_templates["top_bottom"]["template"]
            response = template.format(**result)
            viz_config = {
                "type": "bar",
                "data": result["data"],
                "title": f"Top {result['n']} {result['dimension']} by {result['metric']}"
            }
            return response, viz_config
        
        elif result_type == "correlation":
            template = self.response_templates["correlation"]["template"]
            response = template.format(**result)
            viz_config = {
                "type": "scatter",
                "data": result["data"],
                "x": result["var1"],
                "y": result["var2"],
                "title": f"Correlation: {result['var1']} vs {result['var2']}"
            }
            return response, viz_config
        
        elif result_type == "summary_stats":
            template = self.response_templates["summary_stats"]["template"]
            response = template.format(**result)
            viz_config = {
                "type": "box",
                "data": result["data"],
                "title": f"{result['metric']} Distribution"
            }
            return response, viz_config
        
        elif result_type == "data_overview":
            template = self.response_templates["data_overview"]["template"]
            response = template.format(**result)
            viz_config = {"type": "overview", "data": result["data"]}
            return response, viz_config
        
        else:
            return "I processed your request but couldn't format the response properly.", {}

def create_visualization(viz_config: Dict, df_data=None):
    """Create visualizations based on configuration"""
    if not viz_config or viz_config.get("type") == "overview":
        if df_data is not None:
            st.dataframe(df_data, use_container_width=True)
        return
    
    try:
        viz_type = viz_config["type"]
        data = viz_config["data"]
        title = viz_config.get("title", "Analysis Result")
        
        if viz_type == "bar":
            if isinstance(data, pd.Series):
                fig = px.bar(x=data.index, y=data.values, title=title)
                fig.update_layout(
                    xaxis_title=data.index.name or "Category",
                    yaxis_title=data.name or "Value"
                )
            else:
                fig = px.bar(data, title=title)
            
        elif viz_type == "line":
            if isinstance(data, pd.Series):
                fig = px.line(x=data.index, y=data.values, title=title)
                fig.update_layout(
                    xaxis_title=data.index.name or "Time",
                    yaxis_title=data.name or "Value"
                )
            else:
                fig = px.line(data, title=title)
        
        elif viz_type == "scatter":
            x_col = viz_config.get("x")
            y_col = viz_config.get("y")
            fig = px.scatter(data, x=x_col, y=y_col, title=title, trendline="ols")
        
        elif viz_type == "box":
            if len(data.columns) == 1:
                fig = px.box(y=data.iloc[:, 0], title=title)
            else:
                fig = px.box(data, title=title)
        
        else:
            st.write("Unsupported visualization type")
            return
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

def pivot_analytics_page():
    """Main chatbot interface"""
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AnalyticsChatbot()
        st.session_state.chat_history = []
        st.session_state.data_loaded = False
    
    # Header
    st.title("🤖 Analytics Intelligence Chatbot")
    st.markdown("*Ask questions about your data in natural language*")
    st.markdown("---")
    
    # Sidebar for data loading
    with st.sidebar:
        st.header("📊 Data Setup")
        
        uploaded_file = st.file_uploader("Upload your data", type=['csv', 'xlsx'])
        
        if uploaded_file and not st.session_state.data_loaded:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.main_df = df
                st.session_state.data_loaded = True
                st.success(f"✅ {df.shape[0]} rows, {df.shape[1]} columns loaded")
                
                # Show data info
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                st.write("**Metrics:**", ", ".join(numeric_cols[:5]))
                st.write("**Categories:**", ", ".join(categorical_cols[:5]))
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # Sample data button
        if st.button("📋 Load Sample Data") and not st.session_state.data_loaded:
            sample_df = pd.DataFrame({
                'Product': ['Electronics', 'Clothing', 'Books', 'Home'] * 50,
                'Sales': np.random.randint(100, 2000, 200),
                'Profit': np.random.randint(20, 500, 200),
                'Region': ['North', 'South', 'East', 'West'] * 50,
                'Month': np.tile(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 34)[:200],
                'Customer_Rating': np.random.uniform(3.0, 5.0, 200)
            })
            st.session_state.main_df = sample_df
            st.session_state.data_loaded = True
            st.success("Sample data loaded!")
            st.rerun()
        
        # Reset data
        if st.session_state.data_loaded:
            if st.button("🔄 Reset Data"):
                st.session_state.data_loaded = False
                if 'main_df' in st.session_state:
                    del st.session_state.main_df
                st.rerun()
    
    # Main chat interface
    if not st.session_state.data_loaded:
        st.info("👆 Please upload your data or load sample data to start chatting!")
        return
    
    df = st.session_state.main_df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, (query, response, viz_config) in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.write(query)
            
            # Assistant response
            with st.chat_message("assistant"):
                st.write(response)
                if viz_config:
                    create_visualization(viz_config)
    
    # Query suggestions
    if not st.session_state.chat_history:
        st.subheader("💡 Try asking:")
        col1, col2 = st.columns(2)
        
        with col1:
            suggestions = [
                f"What are the total {numeric_cols[0] if numeric_cols else 'sales'} by {categorical_cols[0] if categorical_cols else 'category'}?",
                f"Show me the top 5 {categorical_cols[0] if categorical_cols else 'items'} by {numeric_cols[0] if numeric_cols else 'value'}",
                f"Compare {numeric_cols[0] if numeric_cols else 'sales'} across {categorical_cols[0] if categorical_cols else 'categories'}"
            ]
        
        with col2:
            more_suggestions = [
                f"What's the average {numeric_cols[0] if numeric_cols else 'value'}?",
                f"Show trends over time" if any('month' in col.lower() or 'date' in col.lower() for col in df.columns) else "Analyze distribution",
                f"Find correlation between {numeric_cols[0]} and {numeric_cols[1]}" if len(numeric_cols) > 1 else "Show data overview"
            ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"sug1_{suggestion[:20]}"):
                st.session_state.current_query = suggestion
        
        for suggestion in more_suggestions:
            if st.button(suggestion, key=f"sug2_{suggestion[:20]}"):
                st.session_state.current_query = suggestion
    
    # Chat input
    query = st.chat_input("Ask me anything about your data...")
    
    # Handle query from suggestions
    if 'current_query' in st.session_state:
        query = st.session_state.current_query
        del st.session_state.current_query
    
    if query:
        # Process query
        with st.spinner("🤔 Analyzing your question..."):
            chatbot = st.session_state.chatbot
            
            # Understand the query
            understanding = chatbot.understand_query(query, df, numeric_cols, categorical_cols)
            
            # Execute the query
            result = chatbot.execute_query(understanding, df)
            
            # Generate response
            response, viz_config = chatbot.generate_response(result)
            
            # Add to chat history
            st.session_state.chat_history.append((query, response, viz_config))
        
        # Rerun to display new message
        st.rerun()


# if __name__ == "__main__":
#     pivot_analytics_page()