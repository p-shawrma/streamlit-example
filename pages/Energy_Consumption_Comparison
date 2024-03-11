import streamlit as st
import psycopg2
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import os
import time
import requests
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydeck as pdk
import datetime
import matplotlib.pyplot as plt
import altair as alt




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
    # print(f"Generated Token: {token}")
else:
    print(f"Failed to execute GraphQL query. Status code: {response.status_code}")
    print(response.text)


# Use the auth_token in the headers for subsequent API requests
headers = {
    "Authorization": f"Bearer {token}"
}

def getReportPagination(uniqueId, start_ts, end_ts):
    graphql_api_url = 'https://log9-api.aquilatrack.com/graphql'

    all_data = []  # List to accumulate data from each day

    # Calculate the total number of days between start_ts and end_ts
    total_days = (end_ts - start_ts) // 86400  # 86400 seconds in a day

    for day in range(total_days + 1):
        # Calculate start and end timestamps for the current day
        current_day_start_ts = start_ts + day * 86400
        current_day_end_ts = min(current_day_start_ts + 86400, end_ts)

        graphql_query = '''
            query GetReportPagination(
                $category: Int!
                $customReportName: String!
                $uniqueId: String!
                $start_ts: String!
                $end_ts: String!
                $clientLoginId: Int!
                $timezone: String!
            ) {
                getReportPagination(
                    category: $category
                    customReportName: $customReportName
                    uniqueId: $uniqueId
                    start_ts: $start_ts
                    end_ts: $end_ts
                    clientLoginId: $clientLoginId
                    timezone: $timezone
                ) {
                    categoryOneFields {
                        dateTime location log9_voltage log9_current log9_soc
                        log9_max_monomer_vol log9_max_vol_cell_no log9_min_monomer_vol
                        log9_monomer_cell_voltage log9_min_vol_cell_no log9_max_monomer_temp
                        log9_min_monomer_temp log9_charging_tube_status log9_discharging_tube_status
                        log9_residual_capacity log9_charge_discharge_cycles log9_error
                        can_raw_data altitude gpsDistance log9_speed log9_drive_mode
                        log9_regen_flag log9_odomoter
                    }
                }
            }
        '''

        variables = {
            'category': 1,
            'customReportName': 'Custom Report',
            'uniqueId': uniqueId,
            'start_ts': str(current_day_start_ts),
            'end_ts': str(current_day_end_ts),
            'clientLoginId': 6487,  # Example clientLoginId, adjust as needed
            'timezone': 'Asia/Calcutta',
        }

        payload = {
            'query': graphql_query,
            'variables': variables
        }

        response = requests.post(graphql_api_url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()['data']['getReportPagination']['categoryOneFields']
            all_data.extend(data)  # Append the day's data
        else:
            print(f"Failed to fetch data for day {day}. Status code: {response.status_code}")
            print(response.text)

    return all_data
    


def GetCategoryOneReport(uniqueId, start_ts, end_ts):
    graphql_api_url = 'https://log9-api.aquilatrack.com/graphql'

    all_data = []  # List to accumulate data from each day

    # Calculate the total number of days between start_ts and end_ts
    total_days = (end_ts - start_ts) // 86400  # 86400 seconds in a day

    for day in range(total_days + 1):
        # Calculate start and end timestamps for the current day
        current_day_start_ts = start_ts + day * 86400
        current_day_end_ts = min(current_day_start_ts + 86400, end_ts)

        graphql_query = '''
            query GetCategoryOneReport(
                $customReportName: String!
                $uniqueId: String!
                $start_ts: String!
                $end_ts: String!
                $clientLoginId: Int!
                $timezone: String!
            ) {
                getCategoryOneReport(
                    customReportName: $customReportName
                    uniqueId: $uniqueId
                    start_ts: $start_ts
                    end_ts: $end_ts
                    clientLoginId: $clientLoginId
                    timezone: $timezone
                ) {
                    uniqueid dateTime location address cumDist speed altitude
                }
            }
        '''

        variables = {
            'customReportName': 'Tracking Report',
            'uniqueId': uniqueId,
            'start_ts': str(current_day_start_ts),
            'end_ts': str(current_day_end_ts),
            'clientLoginId': 6487,  # Example clientLoginId, adjust as needed
            'timezone': 'Asia/Calcutta',
        }

        payload = {
            'query': graphql_query,
            'variables': variables
        }

        response = requests.post(graphql_api_url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()['data']['getCategoryOneReport']
            all_data.extend(data)  # Append the day's data
        else:
            print(f"Failed to fetch data for day {day}. Status code: {response.status_code}")
            print(response.text)

    return all_data



# Function to map selected products to their corresponding values
def get_product_values(product):
    mapping = {
        '12_KW_4W': {'drag_coefficient': 0.7, 'frontal_area': 3.6, 'weight': 1700, 'battery_capacity': 11.77},
        '5.8 KW': {'drag_coefficient': 0.5, 'frontal_area': 3, 'weight': 1030, 'battery_capacity': 5.8},
        '2 KW': {'drag_coefficient': 0.9, 'frontal_area': 0.6, 'weight': 286, 'battery_capacity': 2.0},
        '7.7 KW': {'drag_coefficient': 0.5, 'frontal_area': 3, 'weight': 1200, 'battery_capacity': 7.7},
    }
    return mapping.get(product, {'drag_coefficient': np.nan, 'frontal_area': np.nan, 'weight': np.nan, 'battery_capacity': np.nan})


def main():
    # if 'last_refresh' not in st.session_state:
    #     st.session_state['last_refresh'] = time.time()
    # # Check if the refresh interval has passed
    # refresh_interval = 300  # seconds (e.g., 300 seconds = 5 minutes)
    # current_time = time.time()
    # if current_time - st.session_state['last_refresh'] > refresh_interval:
    #     st.session_state['last_refresh'] = current_time
    #     st.cache_data.clear()
    #     st.experimental_rerun()
        
    df_main, df_tel = get_data()
    df = df_main  # Use df for data from pulkit_main_telematics table
    df2 = df_tel  # Use df2 for data from pulkit_telematics_table
 
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        df_main['date'] = pd.to_datetime(df_main['date'], errors='coerce')
        
        # Calculate the maximum and minimum dates in the dataset
        max_date = df_main['date'].max()
        min_date = df_main['date'].min()
        
        # Calculate the start date for the last 7 days range
        start_date_last_7_days = max_date - pd.Timedelta(days=2)
        
        # Correcting the previous oversight:
        # Ensure the date_range variable correctly uses the last 7 days as the default value
        # and that min_value and max_value parameters are correctly set to define the allowable date range.
        date_range = st.date_input(
            'Select Date Range', 
            value=[start_date_last_7_days, max_date],  # Sets default range to last 7 days
            min_value=min_date,  # The earliest date a user can select
            max_value=max_date   # The latest date a user can select
        )
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
                selected_products = st.selectbox('Product', [""] + products)
    
                # Filter dataframe by product if selected
                if selected_products:
                    df_filtered = df_filtered[df_filtered['product'] == selected_products]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['product'] == selected_products]
    
                # City filter
                cities = df_filtered['deployed_city'].dropna().unique().tolist()
                selected_cities = st.multiselect('City', cities)
    
                # Filter dataframe by city if selected
                if selected_cities:
                    df_filtered = df_filtered[df_filtered['deployed_city'].isin(selected_cities)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['deployed_city'].isin(selected_cities)]
                       
                reg_nos = df_filtered['reg_no'].dropna().unique().tolist()
                selected_reg_nos = st.multiselect('Registration Number', reg_nos, default=[])
                
                # Enforce maximum selection limit of 5
                max_selection = 5
                if len(selected_reg_nos) > max_selection:
                    st.warning(f'Please select no more than {max_selection} registration numbers.')
                    selected_reg_nos = selected_reg_nos[:max_selection]  # Keep only the first 5 selections
                
                # Filter dataframe by registration number if any are selected
                if selected_reg_nos:
                    df_filtered = df_filtered[df_filtered['reg_no'].isin(selected_reg_nos)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['reg_no'].isin(selected_reg_nos)]
        
        
        st.markdown("### Compare Vehicles")
        run_button = st.button("Run")

        st.markdown("### Cache Management")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.experimental_rerun()
        
    if run_button:
        if not df_filtered.empty:
            # Remove rows where telematics_number is None
            df_filtered = df_filtered.dropna(subset=['telematics_number'])
            
            # Add "it_" prefix to telematics_number
            df_filtered['uniqueId'] = 'it_' + df_filtered['telematics_number'].astype(str)
            
            # Group by  and reg_no and get unique combinations
            unique_combinations = df_filtered[['uniqueId', 'reg_no']].dropna().drop_duplicates()

            telematics_number = df_filtered['uniqueId']
            start_date = date_range[0]
            end_date = date_range[1]
            
            # Convert start_date and end_date to datetime.datetime objects
            start_date = datetime.datetime.combine(start_date, datetime.time.min)
            end_date = datetime.datetime.combine(end_date, datetime.time.max)
    
        
        # st.markdown("## Vehicles")
        # st.dataframe(unique_combinations, height=300)
          
        # Remove the "primary_id" column from df_filtered_tel
        df_filtered_tel_without_primary_id = df_filtered_tel.drop(columns=['primary_id'])
        
        df_filtered_without_primary_id = df_filtered.drop(columns=['primary_id'])
        
        # Display the "Day Wise Summary" DataFrame without the "primary_id" column
        st.markdown("## Day Wise Summary")
        st.dataframe(df_filtered_tel_without_primary_id, height=300)  
        

        # Ensure 'date' is in datetime format without time and convert to string
        df_filtered_without_primary_id['date'] = pd.to_datetime(df_filtered_without_primary_id['date']).dt.date.astype(str)
        
        # Remove duplicate rows based on all columns
        df_filtered_without_primary_id = df_filtered_without_primary_id.drop_duplicates()
        
        # Create the scatter plot chart
        scatter = alt.Chart(df_filtered_without_primary_id).mark_point().encode(
            x=alt.X('date:N', axis=alt.Axis(labelAngle=-90)),  # Treat 'date' as nominal and rotate x-axis labels
            y=alt.Y('predicted_range:Q'),  # Quantitative scale for y-axis
            color=alt.Color('reg_no:N', legend=alt.Legend(title="Registration Number")),  # Color by reg_no
            tooltip=['date:N', 'predicted_range:Q', 'reg_no:N']  # Tooltip shows date, predicted_range, and reg_no
        )
        
        # Create the text labels for data points
        text = scatter.mark_text(
            align='left',
            baseline='middle',
            dx=7,  # Nudge text to right so it doesn't overlap with points
            dy=-10  # Nudge text up to avoid overlap with points below
        ).encode(
            text='predicted_range:Q',  # The field you want to display as text
            color=alt.Color('reg_no:N', legend=None)  # Use the same color encoding but without legend for text
        )
        
        # Combine scatter plot and text labels
        chart = scatter + text
        
        # Set the properties for the combined chart
        chart = chart.properties(width=600, height=500)
        
        # Display the scatter plot with data labels in Streamlit
        st.altair_chart(chart, use_container_width=True)
        
        # # Create a bar chart
        # bar_chart = alt.Chart(df_filtered_without_primary_id).mark_bar().encode(
        #     x=alt.X('reg_no:N', sort=None),  # Use 'reg_no' for the X-axis
        #     y=alt.Y('predicted_range:Q', axis=alt.Axis(title='Predicted Range')),  # Use 'predicted_range' for the Y-axis
        #     color='reg_no:N',  # Color bars by 'reg_no' for distinction
        #     column=alt.Column('date:N', header=alt.Header(title='Date')),  # Group bars by 'date'
        #     tooltip=['date:N', 'reg_no:N', 'predicted_range:Q']  # Show these fields in tooltip
        # )
        # bar_chart = bar_chart.properties(width=300, height=500)
        
        # # Display the combined chart with data labels in Streamlit
        # st.altair_chart(bar_chart, use_container_width=False)
  

        # Example DataFrame or list of uniqueIds
        unique_ids = unique_combinations['uniqueId'].unique()  # Replace df with your actual DataFrame variable
        
        all_categoryOneData = []  # This will store the results for all uniqueIds
        all_paginationData = []  # This will store the results for all uniqueIds

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        for uniqueId in unique_ids:
            uniqueId_str = str(uniqueId)  # Ensure uniqueId is a string
            
            # Fetch data from the API for each uniqueId
            try:
                result_category_one = GetCategoryOneReport(uniqueId_str, start_ts, end_ts)
                result_pagination = getReportPagination(uniqueId_str, start_ts, end_ts)
                
                # Convert the results to DataFrames
                result_pagination_df = pd.DataFrame(result_pagination)
                result_category_one_df = pd.DataFrame(result_category_one)
                
                # Add a column for uniqueId in each DataFrame
                result_pagination_df['uniqueId'] = uniqueId_str
                result_category_one_df['uniqueId'] = uniqueId_str
                
                # Append the DataFrame itself to the list, not the dictionary
                all_paginationData.append(result_pagination_df)
                all_categoryOneData.append(result_category_one_df)
                
            except Exception as e:
                print(f"An error occurred while fetching data for {uniqueId_str}: {e}")
        
        df_pagination = pd.concat(all_paginationData, ignore_index=True)
        df_categoryOne = pd.concat(all_categoryOneData, ignore_index=True)

        # # Set ['uniqueId', 'dateTime'] as the index for df_pagination
        # df_pagination.set_index(['uniqueId', 'dateTime'], inplace=True)
        
        # # Set ['uniqueId', 'dateTime'] as the index for df_categoryOne
        # df_categoryOne.set_index(['uniqueId', 'dateTime'], inplace=True)
        
        df_pagination['dateTime'] = pd.to_numeric(df_pagination['dateTime'], errors='coerce')
        df_categoryOne['dateTime'] = pd.to_numeric(df_categoryOne['dateTime'], errors='coerce')
        
        # Convert 'dateTime' from Unix timestamp to datetime in UTC
        df_pagination['dateTime2'] = pd.to_datetime(df_pagination['dateTime'], unit='s')
        df_categoryOne['dateTime2'] = pd.to_datetime(df_categoryOne['dateTime'], unit='s')
        
        # Extract just the date part into a new column
        df_pagination['date'] = df_pagination['dateTime2'].dt.date
        df_categoryOne['date'] = df_categoryOne['dateTime2'].dt.date

        # Define the desired column order for df_pagination
        pagination_columns_order = [
            'uniqueId', 'dateTime', 'dateTime2', 'date', 'location', 'log9_voltage',
            'log9_current', 'log9_soc', 'log9_max_monomer_vol', 'log9_max_vol_cell_no',
            'log9_min_monomer_vol', 'log9_monomer_cell_voltage', 'log9_min_vol_cell_no',
            'log9_max_monomer_temp', 'log9_min_monomer_temp', 'log9_charging_tube_status',
            'log9_discharging_tube_status', 'log9_residual_capacity', 'log9_charge_discharge_cycles',
            'log9_error', 'can_raw_data', 'altitude', 'gpsDistance'
        ]
        
        # Reorder the columns in df_pagination
        df_pagination = df_pagination.reindex(columns=pagination_columns_order)
        # st.markdown("## Custom Report Data")
        # st.dataframe(df_pagination, height=300)

        # Reset index if 'uniqueId' and 'dateTime' are indexes
        df_categoryOne = df_categoryOne.reset_index()
        df_pagination = df_pagination.reset_index()

        # Define the columns to be merged from df_pagination to df_categoryOne
        columns_to_merge = [
            'uniqueId',
            'dateTime',
            'log9_voltage', 
            'log9_current', 
            'log9_soc', 
            'log9_monomer_cell_voltage',
            'log9_max_monomer_vol', 
            'log9_max_vol_cell_no', 
            'log9_min_monomer_vol', 
            'log9_min_vol_cell_no'
        ]
        
        # Perform the left join
        df_categoryOne = pd.merge(
            df_categoryOne,
            df_pagination[columns_to_merge],
            how='left',
            on=['uniqueId', 'dateTime']
        )
        
        # Define the columns to be merged from df_pagination to df_categoryOne
        columns_to_merge = [
            'uniqueId',
            'reg_no',
        ]
        
        # Perform the left join
        df_categoryOne = pd.merge(
            df_categoryOne,
            unique_combinations[columns_to_merge],
            how='left',
            on=['uniqueId']
        )
        
        # Define the desired column order for df_categoryOne
        categoryOne_columns_order = [
            'uniqueId','reg_no', 'dateTime', 'dateTime2', 'date', 'location', 'address',
            'cumDist', 'speed', 'altitude', 'log9_voltage', 'log9_current','log9_soc','log9_monomer_cell_voltage', 
            'log9_max_monomer_vol', 'log9_max_vol_cell_no', 'log9_min_monomer_vol', 
            'log9_min_vol_cell_no'
        ]
        
        # Reorder the columns in df_categoryOne
        df_categoryOne = df_categoryOne.reindex(columns=categoryOne_columns_order)
        # st.markdown("## Tracking Data")
        # st.dataframe(df_categoryOne, height=300)

        # df_categoryOne.set_index(['uniqueId', 'dateTime'], inplace=True)

        df_all_processed = pd.DataFrame()

        for unique_id in df_categoryOne['uniqueId'].unique():
            df_sub = df_categoryOne[df_categoryOne['uniqueId'] == unique_id].copy()

            # Sort the dataframe by 'dateTime' in ascending order
            df_sub = df_sub.sort_values(by='dateTime')
            
            # Now you can calculate the difference in seconds (delta_time) between each row and the previous row
            df_sub['delta_time'] = df_sub['dateTime2'].diff().dt.total_seconds().fillna(0)
            df_sub['delta_time'].iloc[0] = 60
            
            # Calculate acceleration
            df_sub['acceleration'] = np.where(
                (df_sub['delta_time'] < 60) ,
                (df_sub['speed'].diff() / 3.6) / df_sub['delta_time'],
                0
            )
            df_sub['acceleration'].iloc[0] = 0  # Set the first row's acceleration to 0
            
            # Calculate the differences needed for the elevation calculation
            df_sub['cumDist_diff'] = df_sub['cumDist'].diff()
            df_sub['altitude_diff'] = df_sub['altitude'].diff()
            
            # Calculate elevation in radians first based on the given conditions
            df_sub['elevation_radians'] = np.where(
                (df_sub['delta_time'] < 60) & 
                (df_sub['cumDist_diff'] > 0) ,
                np.arctan(df_sub['altitude_diff'] / (df_sub['cumDist_diff'] * 1000)),
                0
            )
            df_sub['cumDist_diff'].iloc[0] = 0  # Set the first row's to 0
            df_sub['altitude_diff'].iloc[0] = 0  # Set the first row's to 0
            
            # Convert elevation from radians to degrees
            df_sub['elevation_degrees'] = np.degrees(df_sub['elevation_radians'])
            
            # Drop the intermediate calculation columns if they are no longer needed
            df_sub.drop(columns=['cumDist_diff', 'altitude_diff'], inplace=True)
            
            # Ensure the first row's elevation is set correctly if necessary
            df_sub['elevation_degrees'].iloc[0] = 0  # Assuming you want to reset the first row's elevation to 0
            
            # Calculate sin(theta) and cos(theta) directly from the elevation angle in radians
            df_sub['sin_theta'] = np.sin(df_sub['elevation_radians'])
            df_sub['cos_theta'] = np.cos(df_sub['elevation_radians'])
    
    
            selected_product = selected_products  # Replace with the actual selected product from the Streamlit selectbox
            product_values = get_product_values(selected_product)
            
            # Add the product-specific values to the dataframe
            df_sub['drag_coefficient'] = product_values['drag_coefficient']
            df_sub['frontal_area'] = product_values['frontal_area']
            df_sub['weight'] = product_values['weight']
            battery_capacity = product_values['battery_capacity']
                    
            # Coefficients and other constants
            stationary_coefficient_of_friction = 0.013
            air_density = 1.20
            g = 9.8  # Acceleration due to gravity in m/s^2
            efficiency = 0.80  # Efficiency factor
            
            # Calculate rolling_coefficient_of_friction
            df_sub['rolling_coefficient_of_friction'] = stationary_coefficient_of_friction * (1 + (df_sub['speed'] / 3.6) / 100)
    
            # Calculating force of rolling resistance
            df_sub['force_rolling_resistance'] = df_sub['weight'] * g * df_sub['rolling_coefficient_of_friction'] * df_sub['cos_theta']
    
            # Calculating force of elevation
            df_sub['force_elevation'] = df_sub['weight'] * g * df_sub['sin_theta']
            
            # Calculating force of drag
            df_sub['force_drag'] = 0.5 * air_density * df_sub['drag_coefficient'] * df_sub['frontal_area'] * ((df_sub['speed'] / 3.6) ** 2)
            
            # Calculating force of acceleration
            df_sub['force_acceleration'] = df_sub['weight'] * df_sub['acceleration']
            
            # Calculating total force of traction
            df_sub['force_traction'] = df_sub['force_acceleration'] + df_sub['force_drag'] + df_sub['force_rolling_resistance'] + df_sub['force_elevation']
            
            # Calculating power of traction
            df_sub['power_traction'] = np.where(
                ((df_sub['delta_time'] < 60) &
                (df_sub['acceleration'] >= 0) ),
                (df_sub['force_traction'] * (df_sub['speed'] / 3.6)) / efficiency,
                0
            )
            
    
          
            # Calculating energy of traction
            df_sub['energy_traction'] = np.where(
                ((df_sub['delta_time'] < 60) &
                (df_sub['acceleration'] >= 0)),
                df_sub['power_traction'] * (df_sub['delta_time'] / 3600),
                0
            )
             
            
            # Calculating energy of rolling resistance
            df_sub['energy_rolling_resistance'] = np.where(
                ((df_sub['delta_time'] < 60) &
                (df_sub['acceleration'] >= 0)),
                (df_sub['force_rolling_resistance'] * (df_sub['speed'] / 3.6)) / efficiency,
                0
            ) * (df_sub['delta_time'] / 3600)
    
            # Calculating energy of rolling resistance
            df_sub['energy_elevation'] = np.where(
                ((df_sub['delta_time'] < 60) &
                (df_sub['acceleration'] >= 0)),
                (df_sub['force_elevation'] * (df_sub['speed'] / 3.6)) / efficiency,
                0
            ) * (df_sub['delta_time'] / 3600)
            
            # Calculating energy of aero drag
            df_sub['energy_aero_drag'] = np.where(
                ((df_sub['delta_time'] < 60) &
                (df_sub['acceleration'] >= 0)),
                (df_sub['force_drag'] * (df_sub['speed'] / 3.6)) / efficiency,
                0
            ) * (df_sub['delta_time'] / 3600)
            
            # Calculating energy of acceleration
            df_sub['energy_acceleration'] = np.where(
                ((df_sub['delta_time'] < 60) &
                (df_sub['acceleration'] >= 0)),
                (df_sub['force_acceleration'] * (df_sub['speed'] / 3.6)) / efficiency,
                0
            ) * (df_sub['delta_time'] / 3600)
            
            # Calculating energy from the battery
            df_sub['energy_actual'] = np.where(
                ((df_sub['delta_time'] < 60) ),
                (df_sub['log9_current'] * df_sub['log9_voltage'] * -1) * (df_sub['delta_time'] / 3600),
                0
            )
    
            # Calculating regenerative energy from the battery
            df_sub['regen_energy_from_battery'] = np.where(
                ((df_sub['log9_current'] > 0) & (df_sub['delta_time'] < 60)),
                (df_sub['log9_current'] * df_sub['log9_voltage'] * -1) * (df_sub['delta_time'] / 3600),
                0
            )
    
            # Calculating energy of traction
            df_sub['energy_calculated'] =  df_sub['energy_traction'] + df_sub['regen_energy_from_battery']
            
            df_all_processed = pd.concat([df_all_processed, df_sub])
            
        # Calculate max and min cumDist for each uniqueId and date
        grouped = df_all_processed.groupby(['uniqueId', 'reg_no', 'date'])['cumDist'].agg([max, min])
        
        st.markdown("## Tracking Data")
        st.dataframe(df_all_processed, height=300)
        
        # Calculate km_travelled
        grouped['km_travelled'] = grouped['max'] - grouped['min']
        
        # Reset index to make 'uniqueId', 'reg_no', and 'date' columns again
        km_travelled = grouped.reset_index()[['uniqueId', 'reg_no', 'date', 'km_travelled']]

        # Create the pivot table with sum of energy columns
        pivot_table = df_all_processed.pivot_table(
            index=['reg_no', 'date'],
            values=['energy_traction','energy_calculated', 'energy_rolling_resistance', 'energy_acceleration', 'energy_aero_drag', 'energy_elevation' , 'energy_actual','regen_energy_from_battery'],
            aggfunc='sum'
        )
        
        # Reset the index to merge with predicted_range data
        pivot_table.reset_index(inplace=True)
        
        # Now, get the predicted_range from df_filtered_without_primary_id
        # Make sure the date column is properly formatted as date
        df_filtered_without_primary_id['date'] = pd.to_datetime(df_filtered_without_primary_id['date']).dt.date
        
        # Drop duplicates based on both 'reg_no' and 'date' columns to ensure a one-to-one merge
        df_filtered_without_primary_id = df_filtered_without_primary_id.drop_duplicates(subset=['reg_no', 'date'])
        
        # Make sure to include 'total_km_travelled' in the merge and now include 'reg_no' in the merge keys
        pivot_table = pd.merge(
            pivot_table,
            km_travelled[['reg_no', 'date', 'km_travelled']],
            on=['reg_no', 'date'],  # Include 'reg_no' here
            how='left'
        )
        
        # Normalize energy columns by 'total_km_travelled' for each row
        for column in ['energy_traction','energy_calculated', 'energy_rolling_resistance', 'energy_elevation', 'energy_acceleration', 'energy_aero_drag','energy_actual','regen_energy_from_battery']:
            pivot_table[column] = pivot_table[column] / pivot_table['km_travelled']
        
        # You might want to handle or replace infinite values or NaNs that result from division by zero
        pivot_table.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Remove rows where 'predicted_range' is NaN
        pivot_table.dropna(subset=['km_travelled'], inplace=True)

        # Identify rows where 'km_travelled' is None or less than 2
        mask = pivot_table['km_travelled'].isna() | (pivot_table['km_travelled'] < 15)
        
        # For rows matching the condition, set specified columns to None
        columns_to_set_none = ['km_travelled','energy_traction', 'energy_calculated', 'energy_rolling_resistance', 'energy_acceleration', 'energy_aero_drag', 'energy_elevation', 'energy_actual', 'regen_energy_from_battery', 'predicted_range']
        pivot_table.loc[mask, columns_to_set_none] = None

        # Check if 'energy_actual' is not zero to avoid division by zero
        pivot_table['predicted_range'] = np.where(
            pivot_table['energy_actual'] != 0,
            battery_capacity * 1000 * 0.95 / pivot_table['energy_actual'],
            np.nan  # Use NaN or an appropriate value for cases where energy_actual is 0
        )
        
        # You might want to handle or replace infinite values or NaNs that result from division by zero
        pivot_table.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Optionally, round the 'predicted_range' to 1 decimal place
        pivot_table['predicted_range'] = pivot_table['predicted_range'].round(1)
        
        pivot_table = pivot_table.round(1)
        
        columns_order = [
            'reg_no',
            'date',
            'predicted_range',
            'km_travelled',
            'energy_actual',
            'energy_calculated',
            'energy_traction',
            'regen_energy_from_battery',
            'energy_rolling_resistance',
            'energy_elevation',
            'energy_acceleration',
            'energy_aero_drag'
        ]

        # Reorder the columns in the pivot table
        pivot_table = pivot_table[columns_order]
        
        # Display the normalized pivot table in the Streamlit app with reordered columns
        st.markdown("## Wh/km distribution")
        st.dataframe(pivot_table)



        # Initialize the boxplot data structure with x-axis and y-axis configurations
        boxplot_data = {
            "xAxis": {
                "type": "category",
                "data": [],
                "axisLabel": {
                    "rotate": 90,  # Rotating x-axis labels by 90 degrees
                    "fontSize": 10
                }
            },
            "yAxis": {"type": "value"},
            "tooltip": {
                "trigger": "item",
                "axisPointer": {"type": "shadow"}
            },
            "series": []  # This will be filled with one series per reg_no
        }
        
        # Find unique reg_nos
        unique_reg_nos = df_all_processed['reg_no'].unique()
        
        # Create a boxplot series for each reg_no
        for reg_no in unique_reg_nos:
            reg_no_data = df_all_processed[(df_all_processed['reg_no'] == reg_no) & (df_all_processed['delta_time'] <= 10) & (df_all_processed['speed'] > 0) ]
            grouped = reg_no_data.groupby('date')['force_acceleration']
            
            # Initialize series data for this reg_no 
            series_data = {
                "name": f"Acceleration force (N) {reg_no}",
                "type": "boxplot",
                "data": []
            }
            
            for name, group in grouped:
                # Calculate the desired percentiles
                percentiles = group.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
                # Add the date to the xAxis if it's not already present
                if name.strftime('%d-%m-%Y') not in boxplot_data["xAxis"]["data"]:
                    boxplot_data["xAxis"]["data"].append(name.strftime('%d-%m-%Y'))
                # Append the percentile data to this series
                series_data["data"].append([
                    percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
                ])
            
            # Add the series for this reg_no to the main boxplot data
            boxplot_data["series"].append(series_data)
        
        # Set the height dynamically based on the number of reg_nos
        chart_height = max(400, 100 * len(unique_reg_nos))
        
        # Render the boxplot using eCharts
        st_echarts(options=boxplot_data, height=f"{chart_height}px")
    
        # # Extract date and hour
        # df_all_processed['hour'] = df_all_processed['dateTime2'].dt.hour
        
        # # Calculate max and min cumDist for each uniqueId, reg_no, date, and hour
        # grouped_hour = df_all_processed.groupby(['uniqueId', 'reg_no', 'date', 'hour'])['cumDist'].agg([max, min])
        
        # # Calculate km_travelled
        # grouped_hour['km_travelled'] = grouped_hour['max'] - grouped_hour['min']
        
        # # Reset index to make 'uniqueId', 'reg_no', 'date', and 'hour' columns again
        # km_travelled_hour = grouped_hour.reset_index()[['uniqueId', 'reg_no', 'date', 'hour', 'km_travelled']]
        
        # # Create the pivot table with sum of energy columns, now including 'hour' in the index
        # pivot_table_hour = df_all_processed.pivot_table(
        #     index=['reg_no', 'date', 'hour'],
        #     values=['energy_traction', 'energy_calculated', 'energy_rolling_resistance', 'energy_acceleration', 'energy_aero_drag', 'energy_elevation', 'energy_actual', 'regen_energy_from_battery'],
        #     aggfunc='sum'
        # )
        
        # # Reset the index to enable merging
        # pivot_table_hour.reset_index(inplace=True)
        
        # # Merge with km_travelled
        # pivot_table_hour = pd.merge(
        #     pivot_table_hour,
        #     km_travelled_hour[['reg_no', 'date', 'hour', 'km_travelled']],
        #     on=['reg_no', 'date', 'hour'],
        #     how='left'
        # )

        
        # # Normalize and clean as before, now also considering 'hour'
        # for column in ['energy_traction', 'energy_calculated', 'energy_rolling_resistance', 'energy_elevation', 'energy_acceleration', 'energy_aero_drag', 'energy_actual', 'regen_energy_from_battery']:
        #     pivot_table_hour[column] = pivot_table_hour[column] / pivot_table_hour['km_travelled']
        
        # pivot_table_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
        # pivot_table_hour.dropna(subset=['km_travelled'], inplace=True)

        # # Identify rows where 'km_travelled' is None or less than 2
        # mask = pivot_table_hour['km_travelled'].isna() | (pivot_table_hour['km_travelled'] < 2)
        
        # # For rows matching the condition, set specified columns to None
        # columns_to_set_none = ['km_travelled','energy_traction', 'energy_calculated', 'energy_rolling_resistance', 'energy_acceleration', 'energy_aero_drag', 'energy_elevation', 'energy_actual', 'regen_energy_from_battery', 'predicted_range']
        # pivot_table_hour.loc[mask, columns_to_set_none] = None
        
        # # Assuming battery_capacity is defined
        # pivot_table_hour['predicted_range'] = np.where(
        #     pivot_table_hour['energy_actual'] != 0,
        #     battery_capacity * 1000 / pivot_table_hour['energy_actual'],
        #     np.nan
        # )
        
        # pivot_table_hour.replace([np.inf, -np.inf], np.nan, inplace=True)
        # pivot_table_hour['predicted_range'] = pivot_table_hour['predicted_range'].round(1)
        # pivot_table_hour = pivot_table_hour.round(1)
        
        # columns_order = [
        #     'reg_no', 'date', 'hour', 'predicted_range', 'km_travelled', 'energy_actual', 'energy_calculated', 'energy_traction', 'regen_energy_from_battery', 'energy_rolling_resistance', 'energy_elevation', 'energy_acceleration', 'energy_aero_drag'
        # ]
        
        # pivot_table_hour = pivot_table_hour[columns_order]
        
        # # Display the pivot table
        # st.markdown("## Wh/km distribution by Date and Hour")
        # st.dataframe(pivot_table_hour)

            

        




















        
    #     # Sort the dataframe by 'dateTime' in ascending order
    #     df_speed_merged = df_speed_merged.sort_values(by='dateTime')
        
    #     # Now you can calculate the difference in seconds (delta_time) between each row and the previous row
    #     df_speed_merged['delta_time'] = df_speed_merged['dateTime2'].diff().dt.total_seconds().fillna(0)
        
    #     # Calculate acceleration
    #     df_speed_merged['acceleration'] = np.where(
    #         (df_speed_merged['log9_current'] < 0) & 
    #         ((df_speed_merged['speed'].diff() / 3.6) / df_speed_merged['delta_time'] > 0),
    #         (df_speed_merged['speed'].diff() / 3.6) / df_speed_merged['delta_time'],
    #         0
    #     )
    #     df_speed_merged['acceleration'].iloc[0] = 0  # Set the first row's acceleration to 0
        
    #     # Calculate the differences needed for the elevation calculation
    #     df_speed_merged['cumDist_diff'] = df_speed_merged['cumDist'].diff()
    #     df_speed_merged['altitude_diff'] = df_speed_merged['altitude'].diff()
        
    #     # Calculate elevation in radians first based on the given conditions
    #     df_speed_merged['elevation_radians'] = np.where(
    #         (df_speed_merged['delta_time'] < 30) & 
    #         (df_speed_merged['cumDist_diff'] > 0),
    #         np.arctan(df_speed_merged['altitude_diff'] / (df_speed_merged['cumDist_diff'] * 1000)),
    #         0
    #     )
    #     df_speed_merged['cumDist_diff'].iloc[0] = 0  # Set the first row's to 0
    #     df_speed_merged['altitude_diff'].iloc[0] = 0  # Set the first row's to 0
        
    #     # Convert elevation from radians to degrees
    #     df_speed_merged['elevation_degrees'] = np.degrees(df_speed_merged['elevation_radians'])
        
    #     # Drop the intermediate calculation columns if they are no longer needed
    #     df_speed_merged.drop(columns=['cumDist_diff', 'altitude_diff'], inplace=True)
        
    #     # Ensure the first row's elevation is set correctly if necessary
    #     df_speed_merged['elevation_degrees'].iloc[0] = 0  # Assuming you want to reset the first row's elevation to 0
        
    #     # Calculate sin(theta) and cos(theta) directly from the elevation angle in radians
    #     df_speed_merged['sin_theta'] = np.sin(df_speed_merged['elevation_radians'])
    #     df_speed_merged['cos_theta'] = 1


    #     selected_product = selected_products  # Replace with the actual selected product from the Streamlit selectbox
    #     product_values = get_product_values(selected_product)
        
    #     # Add the product-specific values to the dataframe
    #     df_speed_merged['drag_coefficient'] = product_values['drag_coefficient']
    #     df_speed_merged['frontal_area'] = product_values['frontal_area']
    #     df_speed_merged['weight'] = product_values['weight']
    #     battery_capacity = product_values['battery_capacity']
                
    #     # Coefficients and other constants
    #     stationary_coefficient_of_friction = 0.012
    #     air_density = 1.20
    #     g = 9.8  # Acceleration due to gravity in m/s^2
    #     efficiency = 0.8  # Efficiency factor
        
    #     # Calculate rolling_coefficient_of_friction
    #     df_speed_merged['rolling_coefficient_of_friction'] = stationary_coefficient_of_friction * (1 + (df_speed_merged['speed'] / 3.6) / 160)

    #     # Calculating force of rolling resistance
    #     df_speed_merged['force_rolling_resistance'] = df_speed_merged['weight'] * g * df_speed_merged['rolling_coefficient_of_friction'] * df_speed_merged['cos_theta']

    #     # Calculating force of elevation
    #     df_speed_merged['force_elevation'] = df_speed_merged['weight'] * g * df_speed_merged['sin_theta']
        
    #     # Calculating force of drag
    #     df_speed_merged['force_drag'] = 0.5 * air_density * df_speed_merged['drag_coefficient'] * df_speed_merged['frontal_area'] * ((df_speed_merged['speed'] / 3.6) ** 2)
        
    #     # Calculating force of acceleration
    #     df_speed_merged['force_acceleration'] = df_speed_merged['weight'] * df_speed_merged['acceleration']
        
    #     # Calculating total force of traction
    #     df_speed_merged['force_traction'] = df_speed_merged['force_acceleration'] + df_speed_merged['force_drag'] + df_speed_merged['force_rolling_resistance'] + df_speed_merged['force_elevation']
        
    #     # Calculating power of traction
    #     df_speed_merged['power_traction'] = np.where(
    #         df_speed_merged['delta_time'] < 30,
    #         (df_speed_merged['force_traction'] * (df_speed_merged['speed'] / 3.6)) / efficiency,
    #         0
    #     )
        

      
    #     # Calculating energy of traction
    #     df_speed_merged['energy_traction'] = np.where(
    #         (df_speed_merged['delta_time'] < 30),
    #         df_speed_merged['power_traction'] * (df_speed_merged['delta_time'] / 3600),
    #         0
    #     )
         
        
    #     # Calculating energy of rolling resistance
    #     df_speed_merged['energy_rolling_resistance'] = np.where(
    #         (df_speed_merged['delta_time'] < 30),
    #         (df_speed_merged['force_rolling_resistance'] * (df_speed_merged['speed'] / 3.6)) / efficiency,
    #         0
    #     ) * (df_speed_merged['delta_time'] / 3600)

    #     # Calculating energy of rolling resistance
    #     df_speed_merged['energy_elevation'] = np.where(
    #         (df_speed_merged['delta_time'] < 30),
    #         (df_speed_merged['force_elevation'] * (df_speed_merged['speed'] / 3.6)) / efficiency,
    #         0
    #     ) * (df_speed_merged['delta_time'] / 3600)
        
    #     # Calculating energy of aero drag
    #     df_speed_merged['energy_aero_drag'] = np.where(
    #         (df_speed_merged['delta_time'] < 30),
    #         (df_speed_merged['force_drag'] * (df_speed_merged['speed'] / 3.6)) / efficiency,
    #         0
    #     ) * (df_speed_merged['delta_time'] / 3600)
        
    #     # Calculating energy of acceleration
    #     df_speed_merged['energy_acceleration'] = np.where(
    #         (df_speed_merged['delta_time'] < 30),
    #         (df_speed_merged['force_acceleration'] * (df_speed_merged['speed'] / 3.6)) / efficiency,
    #         0
    #     ) * (df_speed_merged['delta_time'] / 3600)
        
    #     # Calculating energy from the battery
    #     df_speed_merged['energy_actual'] = np.where(
    #         (df_speed_merged['log9_current'] < 0) & (df_speed_merged['delta_time'] < 30),
    #         (df_speed_merged['log9_current'] * df_speed_merged['log9_voltage'] * -1) * (df_speed_merged['delta_time'] / 3600),
    #         0
    #     )

    #     # Calculating regenerative energy from the battery
    #     df_speed_merged['regen_energy_from_battery'] = np.where(
    #         (df_speed_merged['log9_current'] > 0) & (df_speed_merged['delta_time'] < 30),
    #         (df_speed_merged['log9_current'] * df_speed_merged['log9_voltage'] * -1) * (df_speed_merged['delta_time'] / 3600),
    #         0
    #     )

    #     # Calculating energy of traction
    #     df_speed_merged['energy_calculated'] =  df_speed_merged['energy_traction'] + df_speed_merged['regen_energy_from_battery']
        
    #     st.markdown("## Custom Report Data")
    #     st.dataframe(df, height=300)

    #     st.markdown("## Tracking Data")
    #     st.dataframe(df_speed_merged, height=300)

    #     # Assuming previous steps for creating pivot_table
        

        
    #     # Melting the DataFrame to long format for easier plotting with Altair
    #     long_format = pivot_table.melt(id_vars=['date'], value_vars=['energy_calculated', 'energy_actual'],
    #                                    var_name='Energy Type', value_name='Energy')
        
    #     # Ensure 'date' is treated as a string for plotting
    #     long_format['date'] = long_format['date'].astype(str)
        
    #     # Creating offset for the x-axis position based on 'Energy Type'
    #     long_format['x_offset'] = long_format['date']   + (
    #         long_format['Energy Type'].map({'energy_calculated': ' (calc.)', 'energy_actual': ' (act.)'})
    #     )

    #     # Creating the grouped bar chart
    #     bar_chart = alt.Chart(long_format).mark_bar().encode(
    #         x=alt.X('x_offset:N', title='Date', axis=alt.Axis(labelAngle=-90)),  # Use 'x_offset' for the X-axis position
    #         y=alt.Y('Energy:Q', title='Energy (Wh/km)'),
    #         color='Energy Type:N',
    #         tooltip=['date:N', 'Energy Type:N', 'Energy:Q']
    #     ).properties(width=alt.Step(15), height=600 )  # Control the width of bars
        
    #     # Define the text to go on top of the bars
    #     text = bar_chart.mark_text(
    #         align='center',
    #         baseline='middle',
    #         dy=-5  # Position text above bars
    #     ).encode(
    #         text=alt.Text('Energy:Q', format='.1f'),  # Format text with 1 decimal
    #         x=alt.X('x_offset:N', title='Date')  # Use 'x_offset' for the X-axis position
    #     )
        
    #     # Combine the bar chart and text labels
    #     final_chart = bar_chart + text
    #     st.markdown("## Actual Wh/km VS Calculated Wh/km")
    #     # Display the chart in Streamlit
    #     st.altair_chart(final_chart, use_container_width=True)

    #     stacked_long_format = pivot_table.melt(id_vars=['date'], 
    #         value_vars=[
    #             'regen_energy_from_battery',
    #             'energy_rolling_resistance',
    #             'energy_elevation',
    #             'energy_acceleration',
    #             'energy_aero_drag'
    #         ],
    #         var_name='Energy Component', value_name='Energy Value')
        
    #     # Ensure 'date' is treated as a string for plotting
    #     stacked_long_format['date'] = stacked_long_format['date'].astype(str)
        
    #     # Creating the stacked bar chart
    #     stacked_bar_chart = alt.Chart(stacked_long_format).mark_bar().encode(
    #         x=alt.X('date:N', title='Date', axis=alt.Axis(labelAngle=-90)),  # Use 'date' for the X-axis position
    #         y=alt.Y('Energy Value:Q', title='Energy (Wh/km)', stack='zero'),  # Stack the energy components
    #         color='Energy Component:N',  # Color bars by energy component
    #         tooltip=['date:N', 'Energy Component:N', 'Energy Value:Q']
    #     ) 
        
        
    #     # Combine the bar chart and text labels
    #     final_stacked_chart = (stacked_bar_chart).properties(
    #         width=alt.Step(40),  # Control the width of bars
    #         height=600  # Increase the height of the chart
    #     )
        
    #     # Display the combined chart with labels in Streamlit
    #     st.markdown("## Stacked Energy Components Distribution")
    #     st.altair_chart(final_stacked_chart, use_container_width=True)

    #     st.markdown("## Current Consumption Profile")
        
    #     # Filter df_filtered for the specified conditions including log9_current < 0
    #     df_c = df[(df['log9_current'] < 0)]
                
    #     # Box plot data preparation for log9_current
    #     if not df_c.empty:
    #             grouped = df_c.groupby('date')['log9_current']
    #             boxplot_data = {
    #                     "xAxis": {
    #                     "type": "category",
    #                     "data": [],
    #                     "axisLabel": {
    #                         "rotate": 90,  # Rotating x-axis labels by 90 degrees
    #                         "fontSize": 10
    #                         }
    #                     },
    #                     "yAxis": {"type": "value"},
    #                     "tooltip": {
    #                         "trigger": "item",
    #                         "axisPointer": {"type": "shadow"}
    #                     },
    #                     "series": [{"name": "Log9 Current", "type": "boxplot", "data": []}]
    #                 }
                    
    #             for name, group in grouped:
    #                 # Compute percentiles for the boxplot; adjust if necessary for log9_current analysis
    #                 percentiles = group.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(1).tolist()
    #                 boxplot_data["xAxis"]["data"].append(name.strftime('%d-%m-%Y'))
    #                 boxplot_data["series"][0]["data"].append([
    #                 percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
    #                         ])
                    
    #             # Display the boxplot
    #             st_echarts(options=boxplot_data, height="500px")

    #     # Filter df_filtered for the specified conditions including log9_current < 0
    #     df_a = df[(df['log9_current'] < 0)]

            
    #     st.markdown("## Voltage Range")
                        
    #     # Box plot data preparation for log9_voltagr
    #     if not df.empty:
    #             grouped = df.groupby('date')['log9_voltage']
    #             boxplot_data = {
    #                     "xAxis": {
    #                     "type": "category",
    #                     "data": [],
    #                     "axisLabel": {
    #                         "rotate": 90,  # Rotating x-axis labels by 90 degrees
    #                         "fontSize": 10
    #                         }
    #                     },
    #                     "yAxis": {"type": "value"},
    #                     "tooltip": {
    #                         "trigger": "item",
    #                         "axisPointer": {"type": "shadow"}
    #                     },
    #                     "series": [{"name": "voltage", "type": "boxplot", "data": []}]
    #                 }
                            
    #             for name, group in grouped:
    #                 # Compute percentiles for the boxplot; adjust if necessary for log9_current analysis
    #                 percentiles = group.quantile([0.01, 0.25, 0.5, 0.75, 1]).round(1).tolist()
    #                 boxplot_data["xAxis"]["data"].append(name.strftime('%d-%m-%Y'))
    #                 boxplot_data["series"][0]["data"].append([
    #                 percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
    #                         ])
                            
    #             # Display the boxplot
    #             st_echarts(options=boxplot_data, height="500px")
            
    #     st.markdown("## Speed Change")
    #     # Filter df_filtered for the specified conditions including log9_current < 0
    #     df_s = df_speed[(df_speed['speed'] > 0)]
        
    #     # Box plot data preparation for log9_altitude
    #     if not df_s.empty:
    #             grouped = df_s.groupby('date')['speed']
    #             boxplot_data = {
    #                     "xAxis": {
    #                     "type": "category",
    #                     "data": [],
    #                     "axisLabel": {
    #                         "rotate": 90,  # Rotating x-axis labels by 90 degrees
    #                         "fontSize": 10
    #                         }
    #                     },
    #                     "yAxis": {"type": "value"},
    #                     "tooltip": {
    #                         "trigger": "item",
    #                         "axisPointer": {"type": "shadow"}
    #                     },
    #                     "series": [{"name": "speed", "type": "boxplot", "data": []}]
    #                 }
                            
    #             for name, group in grouped:
    #                 # Compute percentiles for the boxplot; adjust if necessary for log9_current analysis
    #                 percentiles = group.quantile([0.1, 0.25, 0.5, 0.75, 0.99]).round(1).tolist()
    #                 boxplot_data["xAxis"]["data"].append(name.strftime('%d-%m-%Y'))
    #                 boxplot_data["series"][0]["data"].append([
    #                 percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
    #                         ])
                            
    #             # Display the boxplot
    #             st_echarts(options=boxplot_data, height="500px")    
    
    else:
        st.write("Please select filters and click run")


if __name__ == "__main__":
    main()
