import streamlit as st
import psycopg2
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import os
from openai import OpenAI
import time
import requests
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydeck as pdk
import datetime 


# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Set page configuration to wide mode and set page title
st.set_page_config(layout="wide", page_title="Vehicle Telematics Dashboard")

# Function to connect to database and get data using psycopg2
@st.cache_data

def get_data():
    conn = psycopg2.connect(
        database="postgres",
        user='postgres.gqmpfexjoachyjgzkhdf',
        password='Change@2015Log9',
        host='aws-0-ap-south-1.pooler.supabase.com',
        port='5432'
    )
    
    # Query for pulkit_main_telematics table
    query_main = "SELECT * FROM pulkit_main_telematics;"
    df_main = pd.read_sql_query(query_main, conn)
    
    # Query for pulkit_telematics_table
    query_tel = "SELECT * FROM pulkit_telematics_table;"
    df_tel = pd.read_sql_query(query_tel, conn)
    
    conn.close()
    
    return df_main.copy(), df_tel.copy()



# Replace with your API authentication details
graphql_api_url = 'https://log9-api.aquilatrack.com/graphql'  # Replace with your actual API endpoint
username = "EV_Demo"
password = "Demo@123"

# Define the GraphQL query with variables
graphql_query = '''
    query GenerateAuthTokenAPI($username: String!, $password: String!) {
        generateAuthTokenAPI(username: $username, password: $password) {
            token
        }
    }'''

# Define the variables to be passed in the query
variables = {
    'username': username,
    'password': password
}

# Create a JSON payload with the GraphQL query and variables
payload = {
    'query': graphql_query,
    'variables': variables
}

# Send the GraphQL query using a POST request
response = requests.post(graphql_api_url, json=payload)

# Check for a successful response and handle the data
if response.status_code == 200:
    data = response.json()
    token = data['data']['generateAuthTokenAPI']['token']
    print(f"Generated Token: {token}")
else:
    print(f"Failed to execute GraphQL query. Status code: {response.status_code}")
    print(response.text)


# Use the auth_token in the headers for subsequent API requests
headers = {
        "Authorization": token
    }

def getReportPagination(uniqueId, start_ts, end_ts):
    # Replace with your API authentication details
    graphql_api_url = 'https://log9-api.aquilatrack.com/graphql'  # Replace with your actual API endpoint

    # Define the GraphQL query with variables
    graphql_query = '''
        query GetReportPagination(
            $category: Int!
            $customReportName: String!
            $uniqueId: String!
            $start_ts: String!
            $end_ts: String!
            $clientLoginId: Int!
            $timezone: String!
            $offset: Int!
        ) {
            getReportPagination(
                category: $category
                customReportName: $customReportName
                uniqueId: $uniqueId
                start_ts: $start_ts
                end_ts: $end_ts
                clientLoginId: $clientLoginId
                timezone: $timezone
                offset: $offset
            ) {
                categoryOneFields {
                    dateTime location log9_voltage log9_current log9_soc log9_max_monomer_vol log9_max_vol_cell_no log9_min_monomer_vol log9_monomer_cell_voltage log9_min_vol_cell_no log9_max_monomer_temp log9_min_monomer_temp log9_charging_tube_status log9_discharging_tube_status log9_residual_capacity log9_charge_discharge_cycles log9_error can_raw_data altitude gpsDistance log9_speed log9_drive_mode log9_regen_flag log9_odomoter
                }
            }
        }
    '''

    # Define the variables to be passed in the query
    variables = {
        'category': 1,
        'customReportName': 'Custom Report',
        'uniqueId': uniqueId,
        'start_ts': start_ts,
        'end_ts': end_ts,
        'clientLoginId': 6487,
        'timezone': 'Asia/Calcutta',
        'offset': 0
    }

    # Create a JSON payload with the GraphQL query and variables
    payload = {
        'query': graphql_query,
        'variables': variables
    }

    # Send the GraphQL query using a POST request
    response = requests.post(graphql_api_url, json=payload,headers=headers)

    # Check for a successful response and handle the data
    if response.status_code == 200:
        data = response.json()
        categoryOneFields = data['data']['getReportPagination']['categoryOneFields']
        return categoryOneFields
    else:
        print(f"Failed to execute GraphQL query. Status code: {response.status_code}")
        print(response.text)
        return None

def main():
    if 'last_refresh' not in st.session_state:
        st.session_state['last_refresh'] = time.time()
    # Check if the refresh interval has passed
    refresh_interval = 300  # seconds (e.g., 300 seconds = 5 minutes)
    current_time = time.time()
    if current_time - st.session_state['last_refresh'] > refresh_interval:
        st.session_state['last_refresh'] = current_time
        st.caching.clear_cache()
        st.experimental_rerun()
        
    df_main, df_tel = get_data()
    df = df_main  # Use df for data from pulkit_main_telematics table
    df2 = df_tel  # Use df2 for data from pulkit_telematics_table
 
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        df_main['date'] = pd.to_datetime(df_main['date'], errors='coerce')
        min_date, max_date = df_main['date'].min(), df_main['date'].max()
        date_range = st.date_input('Select Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)

        df_filtered = df_main
        df_filtered_tel = df_tel
        
        if len(date_range) == 2 and date_range[0] and date_range[1]:
                # ... [your existing data filtering code]
                df_filtered = df_filtered[(df_filtered['date'] >= pd.Timestamp(date_range[0])) & (df_filtered['date'] <= pd.Timestamp(date_range[1]))]
                
                df_filtered_tel['end_date'] = pd.to_datetime(df_filtered_tel['end_date'])

                df_filtered_tel = df_filtered_tel[(df_filtered_tel['end_date'] >= pd.Timestamp(date_range[0])) & (df_filtered_tel['end_date'] <= pd.Timestamp(date_range[1]))]
    
                # Customer Name filter
                partner_ids = df_filtered['partner_id'].dropna().unique().tolist()
                selected_partner_ids = st.multiselect('Customer Name', partner_ids)
    
                # Filter dataframe by customer name if selected
                if selected_partner_ids:
                    df_filtered = df_filtered[df_filtered['partner_id'].isin(selected_partner_ids)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['partner_id'].isin(selected_partner_ids)]
    
                # Product filter
                products = df_filtered['product'].dropna().unique().tolist()
                selected_products = st.multiselect('Product', products)
    
                # Filter dataframe by product if selected
                if selected_products:
                    df_filtered = df_filtered[df_filtered['product'].isin(selected_products)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['product'].isin(selected_products)]
    

    
                # City filter
                cities = df_filtered['deployed_city'].dropna().unique().tolist()
                selected_cities = st.multiselect('City', cities)
    
                # Filter dataframe by city if selected
                if selected_cities:
                    df_filtered = df_filtered[df_filtered['deployed_city'].isin(selected_cities)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['deployed_city'].isin(selected_cities)]
                       
                # Registration Number filter
                reg_nos = df_filtered['reg_no'].dropna().unique().tolist()
                selected_reg_no = st.selectbox('Registration Number', [""] + reg_nos)  # Add an empty option
        
                # Filter dataframe by registration number if selected
                if selected_reg_no:
                    df_filtered = df_filtered[df_filtered['reg_no'] == selected_reg_no]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['reg_no'] == selected_reg_no]

        
    # Display the filtered dataframe below the charts

    # Create a three-column layout
    col1, col2, col3 = st.columns(3)

    # Display the "Vehicles" dataframe in the first column

    if selected_reg_no:
        if not df_filtered.empty:
            # Remove rows where telematics_number is None
            df_filtered = df_filtered.dropna(subset=['telematics_number'])
            
            # Add "it_" prefix to telematics_number
            df_filtered['telematics_number_with_prefix'] = 'it_' + df_filtered['telematics_number'].astype(str)
            
            # Group by telematics_number_with_prefix and reg_no and get unique combinations
            unique_combinations = df_filtered[['telematics_number_with_prefix', 'reg_no']].dropna().drop_duplicates()

            telematics_number = df_filtered['telematics_number_with_prefix']
            start_date = date_range[0]
            end_date = date_range[1]
            
            # Convert start_date and end_date to datetime.datetime objects
            start_date = datetime.datetime.combine(start_date, datetime.time.min)
            end_date = datetime.datetime.combine(end_date, datetime.time.max)
    
        with col1:
            st.markdown("## Vehicles")
            st.dataframe(unique_combinations, height=300)
          
        # Remove the "primary_id" column from df_filtered_tel
        df_filtered_tel_without_primary_id = df_filtered_tel.drop(columns=['primary_id'])
        telematics_number = df_filtered['telematics_number_with_prefix']
        
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        # Fetch data from the API
        result = getReportPagination(uniqueId, start_ts, end_ts)
            if result:
                # Convert the result into a Pandas DataFrame
                df = pd.DataFrame(result)
            
            
        # Display the "Day Wise Summary" DataFrame without the "primary_id" column
        st.markdown("## Day Wise Summary")
        st.dataframe(df_filtered_tel_without_primary_id, height=300)  
        
        st.markdown("## API Data")
        st.dataframe(df, height=300)
if __name__ == "__main__":
    main()
