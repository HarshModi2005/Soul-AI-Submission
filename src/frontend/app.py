"""
Streamlit frontend for Named Entity Recognition (NER) System.

This module provides a user-friendly web interface for the NER API, featuring:
- Text input for entity analysis
- Visual entity highlighting in results
- Tabular view of extracted entities
- Statistical analysis with visualizations
- API status monitoring and diagnostics
- Error handling with detailed debug information

The UI is designed to be responsive and informative, with multiple ways to
visualize and understand the extracted entities.
"""

import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import base64
from collections import Counter
import plotly.express as px
import os

# -----------------------------------------------------------------------------
# API Connectivity Functions
# -----------------------------------------------------------------------------

def check_api_status():
    """
    Check if the API is accessible and responding.
    
    Attempts to connect to the API's health endpoint and verify its status.
    
    Returns:
        tuple: (bool success, dict/str response)
            - success: True if API is online and responding
            - response: API status info if successful, error message if not
    """
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API returned status code {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------

# Set page configuration for better UI/UX
st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# API Configuration
# -----------------------------------------------------------------------------

# Load API settings from environment variables
API_URL = os.getenv("API_URL", "https://soul-ai-ner-backend.onrender.com")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password123")

def get_auth_header():
    """
    Create HTTP Basic Authentication header for API requests.
    
    Encodes credentials according to the Basic Auth standard.
    
    Returns:
        dict: Authorization header dictionary
    """
    auth_str = base64.b64encode(f"{API_USERNAME}:{API_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {auth_str}"}

# -----------------------------------------------------------------------------
# API Interaction Functions
# -----------------------------------------------------------------------------

def call_ner_api(text):
    """
    Call the NER API and return the results.
    
    Makes a POST request to the prediction endpoint with error handling
    and detailed logging for troubleshooting.
    
    Args:
        text (str): Text to analyze for named entities
        
    Returns:
        dict: API response containing entities and metadata, or None on error
    """
    try:
        # Log request details for debugging
        st.session_state["last_request"] = {
            "url": f"{API_URL}/predict",
            "headers": {"Content-Type": "application/json"},
            "payload": {"text": text}
        }
        
        # Make request with timeout and proper error handling
        response = requests.post(
            f"{API_URL}/predict", 
            headers={**get_auth_header(), "Content-Type": "application/json"},
            json={"text": text},
            timeout=60  # Set a reasonable timeout for model inference
        )
        
        # Log response details for diagnostics
        st.session_state["last_response"] = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": response.text[:500] + "..." if len(response.text) > 500 else response.text
        }
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        
        # Add debug information in an expander for troubleshooting
        with st.expander("Debug Information"):
            if "last_request" in st.session_state:
                st.subheader("Last Request")
                st.json(st.session_state["last_request"])
            if "last_response" in st.session_state:
                st.subheader("Last Response")
                st.json(st.session_state["last_response"])
        
        return None

def call_test_api(text):
    """
    Call the test API endpoint as a fallback.
    
    This lighter endpoint doesn't load the full model, making it useful
    for testing connectivity or when the main API is experiencing issues.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sample entity results, or None on error
    """
    try:
        response = requests.post(
            f"{API_URL}/test-predict", 
            headers={**get_auth_header(), "Content-Type": "application/json"},
            json={"text": text},
            timeout=10  # Short timeout for test endpoint
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Test API Error: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def highlight_entities(text, entities):
    """
    Highlight entities in the original text with HTML.
    
    Creates visual highlighting of detected entities using color-coded
    HTML markup, with different colors for different entity types.
    
    Args:
        text (str): Original text
        entities (list): List of entity dictionaries
        
    Returns:
        str: HTML-formatted text with highlighted entities
    """
    # Sort entities by start position in reverse order to avoid index issues
    # when inserting HTML tags alters string positions
    entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    result = text
    # Color mapping for different entity types
    colors = {
        "PERSON": "#8ef",
        "ORG": "#faa",
        "LOC": "#afa",
        "GPE": "#afa",  # Same as LOC for BERT
        "DATE": "#e9e",
        "TIME": "#f8d",
        "MONEY": "#ad5",
        "PERCENT": "#baa",
        "PRODUCT": "#5cd",
        "EVENT": "#faf", 
        "WORK_OF_ART": "#fab",
        "FAC": "#d8f",   # FACILITY in BERT
        "NORP": "#fed",  # Nationalities, religions, political groups
        "LANGUAGE": "#adf",
        "LAW": "#dfe",
        # Add any other entity types your BERT model uses
        "MISC": "#eee",
        "PER": "#8ef",   # Alternative for PERSON
        "B-PERSON": "#8ef",  # For BIO tagging scheme
        "I-PERSON": "#8ef",
        "B-ORG": "#faa",
        "I-ORG": "#faa"
    }
    
    for entity in entities_sorted:
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
        entity_text = entity["text"]
        
        # Get color for this entity type (default gray if not in our map)
        color = colors.get(label, "#ddd")
        
        # Create the HTML for the highlighted entity with tooltip
        highlighted = f'<mark style="background-color: {color};" title="{label}">{entity_text}</mark>'
        
        # Insert the highlighted entity into the text
        result = result[:start] + highlighted + result[end:]
    
    return result

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

# Title and description
st.title("Named Entity Recognition")
st.markdown("Extract entities such as people, organizations, locations, and more from your text.")

# API Status sidebar
with st.sidebar:
    st.header("API Status")
    if st.button("Check API Connection"):
        status, info = check_api_status()
        if status:
            st.success("✅ API is online")
            st.json(info)
        else:
            st.error(f"❌ API unreachable: {info}")

# Text input area
text_input = st.text_area(
    "Enter text for entity recognition", 
    height=200,
    placeholder="Enter text here... (e.g., 'Apple is looking at buying U.K. startup for $1 billion')",
    help="Type or paste text that you want to analyze for named entities"
)

# Process button
if st.button("Recognize Entities", type="primary"):
    if text_input:
        with st.spinner("Processing..."):
            # Try test endpoint first to avoid memory issues
            # This provides a faster response for basic testing
            result = call_test_api(text_input)
            
            # If test endpoint fails, try the full model endpoint
            if result is None:
                st.warning("Test endpoint failed, trying full model...")
                result = call_ner_api(text_input)
            
            if result:
                entities = result["entities"]
                
                # Create tabs for different views of the results
                tab1, tab2, tab3 = st.tabs(["Highlighted Text", "Entity List", "Statistics"])
                
                with tab1:
                    # Display highlighted text with color-coded entities
                    if entities:
                        highlighted_text = highlight_entities(text_input, entities)
                        st.markdown(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'>{highlighted_text}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No entities found in the text.")
                
                with tab2:
                    # Display entities as a sortable, filterable table
                    if entities:
                        df = pd.DataFrame(entities)
                        df = df.rename(columns={"text": "Entity", "label": "Type", "start": "Start", "end": "End"})
                        df = df[["Entity", "Type", "Start", "End"]]  # Reorder columns
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No entities found in the text.")
                
                with tab3:
                    # Entity statistics and visualizations
                    if entities:
                        # Count entities by type for analysis
                        entity_types = [entity["label"] for entity in entities]
                        type_counts = Counter(entity_types)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create an interactive bar chart
                            chart_data = pd.DataFrame({
                                "Entity Type": list(type_counts.keys()),
                                "Count": list(type_counts.values())
                            })
                            
                            fig = px.bar(
                                chart_data, 
                                x="Entity Type", 
                                y="Count", 
                                title="Entity Types Distribution",
                                color="Entity Type"
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            # Summary statistics in metric displays
                            st.metric("Total Entities", len(entities))
                            st.metric("Entity Types", len(type_counts))
                            
                            # Most common entity type
                            if type_counts:
                                most_common = type_counts.most_common(1)[0]
                                st.metric("Most Common Entity", f"{most_common[0]} ({most_common[1]} occurrences)")
                    else:
                        st.info("No entities found to generate statistics.")
    else:
        st.warning("Please enter some text first.")

# -----------------------------------------------------------------------------
# Reference Information
# -----------------------------------------------------------------------------

# Add explanation of entity types at the bottom
with st.expander("Entity Type Descriptions"):
    st.markdown("""
    | Entity Type | Description |
    | --- | --- |
    | PERSON | People, including fictional characters |
    | ORG | Companies, agencies, institutions |
    | GPE | Geopolitical entity (countries, cities, states) |
    | LOC | Non-GPE locations, mountain ranges, bodies of water |
    | DATE | Dates or periods |
    | TIME | Times smaller than a day |
    | MONEY | Monetary values, including unit |
    | PRODUCT | Objects, vehicles, foods, etc. (not services) |
    | EVENT | Named hurricanes, battles, wars, sports events, etc. |
    """)

# Footer
st.markdown("---")
st.markdown("NER Model powered by BERT and served via FastAPI")