import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import uuid
import json
import os

# Initialize session state
def init_session_state():
    if 'charts' not in st.session_state:
        st.session_state.charts = {
            'default_charts': {},
            'custom_charts': {}
        }
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Chart Studio"

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

# Helper functions
def get_columns_by_type(df, col_type=None):
    """Helper function to get columns by type (numeric, categorical, or all)."""
    if df is None:
        return []
    if col_type == 'numeric':
        return df.select_dtypes(include=['number']).columns.tolist()
    elif col_type == 'categorical':
        return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    else:
        return df.columns.tolist()

def save_charts():
    """Save charts to session state."""
    if 'charts' in st.session_state:
        st.success("Charts saved successfully!")

CHART_COMPATIBILITY = {
    'bar': {
        'required': ['x', 'y'],
        'optional': ['color', 'pattern_shape', 'facet_row', 'facet_col'],
        'agg_required': True
    },
    'line': {
        'required': ['x', 'y'],
        'optional': ['color', 'line_dash', 'facet_row', 'facet_col'],
        'agg_required': True
    },
    'scatter': {
        'required': ['x', 'y'],
        'optional': ['color', 'size', 'symbol', 'facet_row', 'facet_col'],
        'agg_required': False
    },
    'pie': {
        'required': ['names'],
        'optional': ['values'],
        'agg_required': True
    },
    'box': {
        'required': ['x', 'y'],
        'optional': ['color'],
        'agg_required': False
    },
    'histogram': {
        'required': ['x'],
        'optional': ['color', 'facet_row', 'facet_col'],
        'agg_required': False
    },
    'heatmap': {
        'required': ['x', 'y', 'z'],
        'optional': [],
        'agg_required': True
    }
}

# Chart rendering functions
def _validate_columns(df, required_cols, chart_type):
    """Validate that required columns exist in DataFrame."""
    missing_cols = [col for col in required_cols if col and col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for {chart_type} chart: {missing_cols}")
        st.info(f"Available columns: {list(df.columns)}")
        return False
    return True

def _render_bar_chart(df, chart_config):
    """Render bar chart."""
    x_col = chart_config.get('x')
    y_col = chart_config.get('y')
    color_col = chart_config.get('color')
    agg_method = chart_config.get('agg', 'sum')

    # Validate required columns
    if not _validate_columns(df, [x_col, y_col], 'bar'):
        return None

    try:
        # Prepare data with aggregation
        if pd.api.types.is_numeric_dtype(df[y_col]):
            if color_col and color_col in df.columns:
                agg_data = df.groupby([x_col, color_col], as_index=False)[y_col].agg(agg_method)
            else:
                agg_data = df.groupby(x_col, as_index=False)[y_col].agg(agg_method)
        else:
            agg_data = df.copy()

        # Create chart
        if color_col and color_col in agg_data.columns:
            fig = px.bar(agg_data, x=x_col, y=y_col, color=color_col, barmode='group')
        else:
            fig = px.bar(agg_data, x=x_col, y=y_col)
        return fig
    except Exception as e:
        st.error(f"Error rendering bar chart: {str(e)}")
        return None

def _render_line_chart(df, chart_config):
    """Render line chart."""
    x_col = chart_config.get('x')
    y_col = chart_config.get('y')
    color_col = chart_config.get('color')
    agg_method = chart_config.get('agg', 'mean')

    # Validate required columns
    if not _validate_columns(df, [x_col, y_col], 'line'):
        return None

    try:
        # Prepare data with aggregation
        if pd.api.types.is_numeric_dtype(df[y_col]):
            if color_col and color_col in df.columns:
                agg_data = df.groupby([x_col, color_col], as_index=False)[y_col].agg(agg_method)
            else:
                agg_data = df.groupby(x_col, as_index=False)[y_col].agg(agg_method)
        else:
            agg_data = df.copy()

        # Create chart
        if color_col and color_col in agg_data.columns:
            fig = px.line(agg_data, x=x_col, y=y_col, color=color_col)
        else:
            fig = px.line(agg_data, x=x_col, y=y_col)
        return fig
    except Exception as e:
        st.error(f"Error rendering line chart: {str(e)}")
        return None

def _render_scatter_chart(df, chart_config):
    """Render scatter chart."""
    x_col = chart_config.get('x')
    y_col = chart_config.get('y')
    color_col = chart_config.get('color')
    size_col = chart_config.get('size')

    # Validate required columns
    if not _validate_columns(df, [x_col, y_col], 'scatter'):
        return None

    try:
        # Create chart
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col if color_col and color_col in df.columns else None,
            size=size_col if size_col and size_col in df.columns else None
        )
        return fig
    except Exception as e:
        st.error(f"Error rendering scatter chart: {str(e)}")
        return None

def _render_pie_chart(df, chart_config):
    """Render pie chart."""
    names_col = chart_config.get('names')
    values_col = chart_config.get('values')
    agg_method = chart_config.get('agg', 'sum')

    # Validate required columns
    if not _validate_columns(df, [names_col], 'pie'):
        return None

    try:
        # Prepare data
        if values_col and values_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[values_col]):
                agg_data = df.groupby(names_col, as_index=False)[values_col].agg(agg_method)
                values = agg_data[values_col]
                names = agg_data[names_col]
            else:
                values = df[names_col].value_counts().values
                names = df[names_col].value_counts().index
        else:
            # Count occurrences if values not provided
            value_counts = df[names_col].value_counts()
            values = value_counts.values
            names = value_counts.index

        # Create chart
        fig = px.pie(names=names, values=values)
        return fig
    except Exception as e:
        st.error(f"Error rendering pie chart: {str(e)}")
        return None

def _render_box_chart(df, chart_config):
    """Render box plot."""
    x_col = chart_config.get('x')
    y_col = chart_config.get('y')
    color_col = chart_config.get('color')

    # Validate required columns
    if not _validate_columns(df, [x_col, y_col], 'box'):
        return None

    try:
        # Create chart
        fig = px.box(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col if color_col and color_col in df.columns else None
        )
        return fig
    except Exception as e:
        st.error(f"Error rendering box chart: {str(e)}")
        return None

def _render_histogram_chart(df, chart_config):
    """Render histogram."""
    x_col = chart_config.get('x')
    color_col = chart_config.get('color')

    # Validate required columns
    if not _validate_columns(df, [x_col], 'histogram'):
        return None

    try:
        # Create chart
        fig = px.histogram(
            df, 
            x=x_col, 
            color=color_col if color_col and color_col in df.columns else None,
            marginal="box"
        )
        return fig
    except Exception as e:
        st.error(f"Error rendering histogram: {str(e)}")
        return None

def _render_heatmap_chart(df, chart_config):
    """Render heatmap."""
    x_col = chart_config.get('x')
    y_col = chart_config.get('y')
    z_col = chart_config.get('z')
    agg_method = chart_config.get('agg', 'mean')

    # Validate required columns
    if not _validate_columns(df, [x_col, y_col, z_col], 'heatmap'):
        return None

    try:
        # Prepare data
        agg_data = df.groupby([x_col, y_col], as_index=False)[z_col].agg(agg_method)
        pivot_data = agg_data.pivot(index=y_col, columns=x_col, values=z_col)
        
        # Create chart
        fig = px.imshow(
            pivot_data,
            labels=dict(x=x_col, y=y_col, color=z_col),
            x=pivot_data.columns,
            y=pivot_data.index
        )
        return fig
    except Exception as e:
        st.error(f"Error rendering heatmap: {str(e)}")
        return None

def render_chart(df, chart_config):
    """
    Renders a chart using Plotly Express based on the chart configuration.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        chart_config (dict): The configuration dictionary for the chart.
    
    Returns:
        plotly.graph_objects.Figure or None: The Plotly figure if rendering is successful, otherwise None.
    """
    # Input validation
    if df is None or df.empty:
        st.warning("Cannot render chart: DataFrame is empty.")
        return None
    
    if not chart_config or not isinstance(chart_config, dict):
        st.warning("Cannot render chart: Invalid chart configuration.")
        return None
    
    chart_type = chart_config.get('type')
    if not chart_type:
        st.warning("Cannot render chart: Chart type not specified.")
        return None

    try:
        fig = None
        
        if chart_type == 'bar':
            fig = _render_bar_chart(df, chart_config)
        elif chart_type == 'line':
            fig = _render_line_chart(df, chart_config)
        elif chart_type == 'scatter':
            fig = _render_scatter_chart(df, chart_config)
        elif chart_type == 'pie':
            fig = _render_pie_chart(df, chart_config)
        elif chart_type == 'box':
            fig = _render_box_chart(df, chart_config)
        elif chart_type == 'histogram':
            fig = _render_histogram_chart(df, chart_config)
        elif chart_type == 'heatmap':
            fig = _render_heatmap_chart(df, chart_config)
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return None

        if fig is None:
            st.warning(f"Failed to create {chart_type} chart. Please check your data and configuration.")
            return None

        # Update layout for better display
        fig.update_layout(
            title=chart_config.get('title', f"{chart_type.capitalize()} Chart"),
            margin=dict(l=50, r=50, t=60, b=50),
            hovermode="closest",
            height=chart_config.get('height', 500),
            width=chart_config.get('width', None)
        )

        return fig
        
    except Exception as e:
        st.error(f"Error rendering {chart_type} chart: {str(e)}")
        st.error("Please check your data types and column names.")
        return None

def validate_chart_config(chart_config, compatibility):
    """
    Validate that all required fields are present in the chart configuration.
    
    Args:
        chart_config: Chart configuration dictionary
        compatibility: Chart type compatibility requirements
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    for required_field in compatibility['required']:
        if required_field not in chart_config or chart_config[required_field] is None:
            return False
    return True

def build_chart_config_ui(df, chart_config, compatibility, numeric_cols, categorical_cols, all_cols, key_prefix=""):
    """
    Build the UI elements for chart configuration.
    
    Args:
        df: DataFrame
        chart_config: Current chart configuration
        compatibility: Chart type compatibility info
        numeric_cols: List of numeric columns
        categorical_cols: List of categorical columns
        all_cols: List of all columns
        key_prefix: Prefix for Streamlit widget keys
    
    Returns:
        Updated chart configuration
    """
    
    # X-Axis configuration
    if 'x' in compatibility['required'] or 'x' in compatibility['optional']:
        x_options = all_cols if chart_config['type'] != 'histogram' else numeric_cols
        if x_options:
            current_x = chart_config.get('x')
            index = x_options.index(current_x) if current_x in x_options else 0
            chart_config['x'] = st.selectbox(
                "X-Axis", 
                x_options, 
                index=index,
                key=f"{key_prefix}x_axis"
            )
        elif 'x' in compatibility['required']:
            st.warning(f"No suitable columns for X-Axis for {chart_config['type']} chart.")
            return chart_config

    # Y-Axis configuration
    if 'y' in compatibility['required'] or 'y' in compatibility['optional']:
        if numeric_cols:
            current_y = chart_config.get('y')
            index = numeric_cols.index(current_y) if current_y in numeric_cols else 0
            chart_config['y'] = st.selectbox(
                "Y-Axis", 
                numeric_cols, 
                index=index,
                key=f"{key_prefix}y_axis"
            )
        elif 'y' in compatibility['required']:
            st.warning(f"No suitable columns for Y-Axis for {chart_config['type']} chart.")
            return chart_config

    # Z-Axis configuration (for heatmaps)
    if 'z' in compatibility['required'] or 'z' in compatibility['optional']:
        if numeric_cols:
            current_z = chart_config.get('z')
            index = numeric_cols.index(current_z) if current_z in numeric_cols else 0
            chart_config['z'] = st.selectbox(
                "Z-Axis (Value)", 
                numeric_cols, 
                index=index,
                key=f"{key_prefix}z_axis"
            )
        elif 'z' in compatibility['required']:
            st.warning(f"No suitable columns for Z-Axis for {chart_config['type']} chart.")
            return chart_config

    # Names configuration (for pie charts)
    if 'names' in compatibility['required'] or 'names' in compatibility['optional']:
        if categorical_cols:
            current_names = chart_config.get('names')
            index = categorical_cols.index(current_names) if current_names in categorical_cols else 0
            chart_config['names'] = st.selectbox(
                "Segment Names", 
                categorical_cols, 
                index=index,
                key=f"{key_prefix}names"
            )
        elif 'names' in compatibility['required']:
            st.warning(f"No suitable columns for Segment Names for {chart_config['type']} chart.")
            return chart_config

    # Values configuration (optional for pie charts)
    if 'values' in compatibility['optional']:
        values_options = [None] + numeric_cols
        current_values = chart_config.get('values')
        index = values_options.index(current_values) if current_values in values_options else 0
        chart_config['values'] = st.selectbox(
            "Segment Values (Optional)", 
            values_options, 
            index=index,
            key=f"{key_prefix}values"
        )

    # Color configuration
    if 'color' in compatibility['optional']:
        color_options = [None] + categorical_cols
        current_color = chart_config.get('color')
        index = color_options.index(current_color) if current_color in color_options else 0
        chart_config['color'] = st.selectbox(
            "Color By (Optional)", 
            color_options, 
            index=index,
            key=f"{key_prefix}color"
        )

    # Size configuration
    if 'size' in compatibility['optional']:
        size_options = [None] + numeric_cols
        current_size = chart_config.get('size')
        index = size_options.index(current_size) if current_size in size_options else 0
        chart_config['size'] = st.selectbox(
            "Size By (Optional)", 
            size_options, 
            index=index,
            key=f"{key_prefix}size"
        )

    # Aggregation configuration
    if compatibility.get('agg_required', False):
        agg_options = ['mean', 'sum', 'count', 'min', 'max']
        current_agg = chart_config.get('agg', 'mean')
        index = agg_options.index(current_agg) if current_agg in agg_options else 0
        chart_config['agg'] = st.selectbox(
            "Aggregation Method", 
            agg_options, 
            index=index,
            key=f"{key_prefix}aggregation"
        )
    else:
        # Remove aggregation if not required for the new chart type
        if 'agg' in chart_config:
            del chart_config['agg']

    return chart_config


def create_new_chart(df):
    """
    Provides UI to create a new chart.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    st.header("Create New Chart")
    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
        return

    chart_types = list(CHART_COMPATIBILITY.keys())
    col1, col2 = st.columns(2)
    
    with col1:
        chart_title = st.text_input("Chart Title", "New Chart")
        chart_type = st.selectbox("Chart Type", chart_types)
        tab_name = st.text_input("Tab Name", "Custom")
    
    with col2:
        chart_config = {
            'id': str(uuid.uuid4()), 
            'title': chart_title, 
            'type': chart_type, 
            'tab': tab_name
        }
        
        compatibility = CHART_COMPATIBILITY[chart_type]
        numeric_cols = get_columns_by_type(df, 'numeric')
        categorical_cols = get_columns_by_type(df, 'categorical')
        all_cols = get_columns_by_type(df)

        # Dynamically add configuration fields
        chart_config = build_chart_config_ui(df, chart_config, compatibility, 
                                           numeric_cols, categorical_cols, all_cols)

    # Chart Preview
    st.markdown("### Chart Preview")
    if validate_chart_config(chart_config, compatibility):
        try:
            fig = render_chart(df, chart_config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to generate chart preview")
        except Exception as e:
            st.error(f"Error generating preview: {str(e)}")
    else:
        st.info("Select required columns to see the chart preview.")

    # Save button
    if st.button("Save Chart", key="save_new_chart"):
        if validate_chart_config(chart_config, compatibility):
            if 'custom_charts' not in st.session_state.charts:
                st.session_state.charts['custom_charts'] = {}
            st.session_state.charts['custom_charts'][chart_config['id']] = chart_config
            save_charts()
            st.success(f"Chart '{chart_title}' saved successfully!")
            st.info(" View your chart in the 'Custom Dashboard' tab!")
            
        else:
            st.error("Please complete all required fields before saving.")

def chart_editor(df):
    """
    Provides UI to edit existing custom charts.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    st.header("Edit Charts")
    
    # Check if we have a selected chart for editing from dashboard
    if 'selected_chart_for_edit' in st.session_state:
        selected_chart_id = st.session_state.selected_chart_for_edit
        del st.session_state.selected_chart_for_edit  # Clear the selection
    else:
        selected_chart_id = None
    
    # Get all charts (both default and custom)
    all_charts = {}
    if 'default_charts' in st.session_state.charts:
        all_charts.update(st.session_state.charts['default_charts'])
    if 'custom_charts' in st.session_state.charts:
        all_charts.update(st.session_state.charts['custom_charts'])
    
    if not all_charts:
        st.info("No charts available to edit. Create some charts first!")
        return

    # Chart selection
    chart_options = {chart_id: f"{config.get('title', 'Untitled')} ({config.get('type', 'Unknown')})" 
                    for chart_id, config in all_charts.items()}
    
    # Use pre-selected chart if coming from dashboard
    if selected_chart_id and selected_chart_id in chart_options:
        default_index = list(chart_options.keys()).index(selected_chart_id)
    else:
        default_index = 0
    
    selected_chart_id = st.selectbox(
        "Select chart to edit", 
        options=list(chart_options.keys()),
        format_func=lambda x: chart_options[x],
        key="chart_selector",
        index=default_index
    )

    if not selected_chart_id:
        return

    # Get the chart configuration
    if selected_chart_id in st.session_state.charts.get('custom_charts', {}):
        chart_config = st.session_state.charts['custom_charts'][selected_chart_id].copy()
        chart_source = 'custom_charts'
    elif selected_chart_id in st.session_state.charts.get('default_charts', {}):
        chart_config = st.session_state.charts['default_charts'][selected_chart_id].copy()
        chart_source = 'default_charts'
    else:
        st.error("Chart not found!")
        return

    st.subheader(f"Editing: {chart_config.get('title', 'Untitled Chart')}")

    # Edit form
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic chart properties
        chart_config['title'] = st.text_input("Chart Title", chart_config.get('title', ''), key="edit_title")
        
        new_chart_type = st.selectbox(
            "Chart Type", 
            list(CHART_COMPATIBILITY.keys()), 
            index=list(CHART_COMPATIBILITY.keys()).index(chart_config.get('type', 'bar')),
            key="edit_chart_type"
        )
        chart_config['type'] = new_chart_type
        
        chart_config['tab'] = st.text_input("Tab Name", chart_config.get('tab', 'Custom'), key="edit_tab")
    
    with col2:
        # Chart-specific configuration
        compatibility = CHART_COMPATIBILITY[new_chart_type]
        numeric_cols = get_columns_by_type(df, 'numeric')
        categorical_cols = get_columns_by_type(df, 'categorical')
        all_cols = get_columns_by_type(df)

        # Build configuration UI for editing
        chart_config = build_chart_config_ui(df, chart_config, compatibility, 
                                           numeric_cols, categorical_cols, all_cols, 
                                           key_prefix="edit_")

    # Chart Preview
    st.markdown("### Updated Chart Preview")
    if validate_chart_config(chart_config, compatibility):
        try:
            fig = render_chart(df, chart_config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to generate chart preview")
        except Exception as e:
            st.error(f"Error generating preview: {str(e)}")
    else:
        st.info("Complete required fields to see the chart preview.")

    # Save changes
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Changes", key="save_chart_changes"):
            if validate_chart_config(chart_config, compatibility):
                st.session_state.charts[chart_source][selected_chart_id] = chart_config
                save_charts()
                st.success(f"Chart '{chart_config['title']}' updated successfully!")
                
            else:
                st.error("Please complete all required fields before saving.")
    
    with col2:
        if st.button("Cancel Changes", key="cancel_chart_changes"):
            st.info("Changes cancelled.")
            st.rerun()
    
    with col3:
        if st.button("View in Dashboard", key="view_in_dashboard"):
            st.session_state.current_page = "Custom Dashboard"
            st.rerun()

def delete_charts():
    """
    Provides UI to delete custom charts.
    """
    st.header("Delete Charts")
    
    # Get all charts
    all_charts = {}
    if 'default_charts' in st.session_state.charts:
        for chart_id, config in st.session_state.charts['default_charts'].items():
            all_charts[chart_id] = f"{config.get('title', 'Untitled')} (Default)"
    if 'custom_charts' in st.session_state.charts:
        for chart_id, config in st.session_state.charts['custom_charts'].items():
            all_charts[chart_id] = f"{config.get('title', 'Untitled')} (Custom)"
    
    if not all_charts:
        st.info("No charts available to delete.")
        return

    # Multi-select for deletion
    charts_to_delete = st.multiselect(
        "Select charts to delete",
        options=list(all_charts.keys()),
        format_func=lambda x: all_charts[x]
    )

    if charts_to_delete:
        st.warning(f"You are about to delete {len(charts_to_delete)} chart(s). This action cannot be undone.")
        
        if st.button(" Delete Selected Charts", type="primary"):
            deleted_count = 0
            for chart_id in charts_to_delete:
                if chart_id in st.session_state.charts.get('custom_charts', {}):
                    del st.session_state.charts['custom_charts'][chart_id]
                    deleted_count += 1
                elif chart_id in st.session_state.charts.get('default_charts', {}):
                    del st.session_state.charts['default_charts'][chart_id]
                    deleted_count += 1
            
            save_charts()
            st.success(f"Successfully deleted {deleted_count} chart(s)!")
            st.rerun()

def chart_studio_page(df):
    st.title(" Chart Studio")

    # Add a refresh button to clear any stuck states
    if st.button(" Refresh Studio"):
        if 'edit_mode' in st.session_state:
            del st.session_state.edit_mode
        if 'selected_chart_for_edit' in st.session_state:
            del st.session_state.selected_chart_for_edit
        st.rerun()

    option = st.radio("Options", ["Create Chart", "Edit Charts", "Delete Charts"])

    if option == "Create Chart":
        create_new_chart(df)
    elif option == "Edit Charts":
        chart_editor(df)
    elif option == "Delete Charts":
        delete_charts()

def custom_dashboard_page(df):
    st.title(" Custom Dashboard")
    
    # Get custom charts
    custom_charts = st.session_state.charts.get('custom_charts', {})
    
    if not custom_charts:
        st.info("No custom charts found. Create some charts first to see them in your dashboard!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create New Chart", help="Go to Chart Studio to create your first chart"):
                st.session_state.current_page = "Chart Studio"
                st.rerun()
        return
    
    # Dashboard controls
    with st.sidebar:
        st.subheader("Dashboard Controls")
        
        # Filter by chart type
        chart_types = list(set([config.get('type', 'Unknown') for config in custom_charts.values()]))
        chart_types.insert(0, 'All')
        selected_type = st.selectbox("Filter by Chart Type", chart_types)
        
        # Filter by tab
        tabs = list(set([config.get('tab', 'Custom') for config in custom_charts.values()]))
        tabs.insert(0, 'All')
        selected_tab = st.selectbox("Filter by Tab", tabs)
        
        # Layout options
        layout_option = st.selectbox("Dashboard Layout", ["Grid (2 columns)", "Grid (3 columns)", "Single Column"])
        
        # Refresh button
        if st.button(" Refresh Dashboard", help="Refresh all charts"):
            st.rerun()
        
        # Export button
        if st.button(" Export Dashboard", help="Export dashboard configuration"):
            export_config = {
                'dashboard_name': 'Custom Dashboard',
                'created_at': pd.Timestamp.now().isoformat(),
                'charts': custom_charts,
                'total_charts': len(custom_charts)
            }
            st.download_button(
                label="Download Configuration",
                data=str(export_config),
                file_name="dashboard_config.json",
                mime="application/json"
            )
    
    # Filter charts based on selections
    filtered_charts = {}
    for chart_id, config in custom_charts.items():
        type_match = selected_type == 'All' or config.get('type') == selected_type
        tab_match = selected_tab == 'All' or config.get('tab') == selected_tab
        
        if type_match and tab_match:
            filtered_charts[chart_id] = config
    
    if not filtered_charts:
        st.warning("No charts match the selected filters.")
        return
    
    # Dashboard statistics
    st.subheader("Dashboard Overview")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Charts", len(filtered_charts))
    
    with stat_col2:
        chart_types_count = len(set([config.get('type', 'Unknown') for config in filtered_charts.values()]))
        st.metric("Chart Types", chart_types_count)
    
    with stat_col3:
        tabs_count = len(set([config.get('tab', 'Custom') for config in filtered_charts.values()]))
        st.metric("Tabs", tabs_count)
    
    with stat_col4:
        if df is not None:
            st.metric("Data Rows", len(df))
    
    st.divider()
    
    # Determine columns based on layout option
    if layout_option == "Grid (2 columns)":
        cols_per_row = 2
    elif layout_option == "Grid (3 columns)":
        cols_per_row = 3
    else:
        cols_per_row = 1
    
    # Group charts by tab for better organization
    charts_by_tab = {}
    for chart_id, config in filtered_charts.items():
        tab = config.get('tab', 'Custom')
        if tab not in charts_by_tab:
            charts_by_tab[tab] = {}
        charts_by_tab[tab][chart_id] = config
    
    # Display charts grouped by tabs
    for tab_name, tab_charts in charts_by_tab.items():
        if len(charts_by_tab) > 1:  # Only show tab headers if there are multiple tabs
            st.subheader(f" {tab_name}")
        
        # Create grid layout for charts
        chart_items = list(tab_charts.items())
        for i in range(0, len(chart_items), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(chart_items):
                    chart_id, config = chart_items[i + j]
                    
                    with cols[j]:
                        # Create a container for each chart
                        with st.container():
                            # Chart header with title and type
                            chart_title = config.get('title', 'Untitled Chart')
                            chart_type = config.get('type', 'Unknown').title()
                            
                            st.markdown(f"#### {chart_title}")
                            st.caption(f"Type: {chart_type} | ID: {chart_id[:8]}...")
                            
                            # Render the chart
                            if df is not None and not df.empty:
                                try:
                                    compatibility = CHART_COMPATIBILITY.get(config.get('type'), {})
                                    if validate_chart_config(config, compatibility):
                                        fig = render_chart(df, config)
                                        if fig:
                                            # Make charts smaller for dashboard view
                                            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.error(" Failed to render chart")
                                    else:
                                        st.warning(" Invalid chart configuration")
                                except Exception as e:
                                    st.error(f" Error: {str(e)[:50]}...")
                            else:
                                st.info(" No data available")
                            
                            # Quick action buttons
                            with st.expander("Chart Options", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(" Edit", key=f"edit_{chart_id}", help="Edit this chart"):
                                        st.session_state.selected_chart_for_edit = chart_id
                                        st.session_state.current_page = "Chart Studio"
                                        st.rerun()
                                with col2:
                                    if st.button(" Delete", key=f"delete_{chart_id}", help="Delete this chart"):
                                        if chart_id in st.session_state.charts['custom_charts']:
                                            del st.session_state.charts['custom_charts'][chart_id]
                                            save_charts()
                                            st.success(f"Deleted '{chart_title}'")
                                            st.rerun()
            
            # Add some spacing between rows
            if i + cols_per_row < len(chart_items):
                st.markdown("---")

# # Main app
# def main():
#     # Initialize session state
#     init_session_state()
    
#     # Create sample data
#     data = {
#         "Date": pd.date_range(start="2023-01-01", periods=100),
#         "Campaign_ID": [str(i) for i in range(1, 101)],
#         "Company": np.random.choice(["Innovate Industries", "Acme Corp", "Beta LLC"], 100),
#         "Campaign_Type": np.random.choice(["Email", "Social Media", "PPC", "Display"], 100),
#         "Target_Audience": np.random.choice(["Men 18-24", "Women 25-34", "All Adults"], 100),
#         "Duration": np.random.choice(["30 days", "60 days", "90 days"], 100),
#         "Channel_Used": np.random.choice(["Google Ads", "Facebook", "LinkedIn"], 100),
#         "Conversion_Rate": np.random.uniform(0.01, 0.15, 100),
#         "Acquisition_Cost": np.random.uniform(5000, 20000, 100),
#         "ROI": np.random.uniform(0.5, 5.0, 100),
#         "Location": np.random.choice(["Chicago", "New York", "Los Angeles"], 100),
#         "Language": np.random.choice(["English", "Spanish", "French"], 100),
#         "Clicks": np.random.randint(100, 5000, 100),
#         "Impressions": np.random.randint(1000, 100000, 100),
#         "Engagement_Score": np.random.uniform(1, 10, 100),
#         "Customer_Segment": np.random.choice(["Health & Wellness", "Tech", "Finance"], 100)
#     }
#     df = pd.DataFrame(data)
#     df["Date"] = pd.to_datetime(df["Date"])
    
#     # Navigation sidebar
#     with st.sidebar:
#         st.title("Navigation")
#         if st.button(" Custom Dashboard"):
#             st.session_state.current_page = "Custom Dashboard"
#             st.rerun()
#         if st.button(" Chart Studio"):
#             st.session_state.current_page = "Chart Studio"
#             st.rerun()
#         st.divider()
#         st.caption(f"Current Page: {st.session_state.current_page}")
    
#     # Display the current page
#     if st.session_state.current_page == "Custom Dashboard":
#         custom_dashboard_page(df)
#     elif st.session_state.current_page == "Chart Studio":
#         chart_studio_page(df)

# if _name_ == "_main_":
#     main()
