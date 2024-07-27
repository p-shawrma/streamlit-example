import streamlit as st
import psycopg2
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import os
import time
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
import pygwalker as pyg
import clickhouse_connect

# ClickHouse connection details
ch_host = 'a84a1hn9ig.ap-south-1.aws.clickhouse.cloud'
ch_user = 'default'
ch_password = 'dKd.Y9kFMv06x'
ch_database = 'landing_zone_telematics'

# Create ClickHouse client
client = clickhouse_connect.get_client(
    host=ch_host,
    user=ch_user,
    password=ch_password,
    database=ch_database,
    secure=True
)

# Set page configuration to wide mode and set page title
st.set_page_config(layout="wide", page_title="Vehicle Telematics Dashboard")

# Mapbox Access Token
# Set Mapbox access token
px.set_mapbox_access_token("pk.eyJ1IjoicC1zaGFybWEiLCJhIjoiY2xzNjRzbTY1MXNodjJsbXUwcG0wNG50ciJ9.v32bwq-wi6whz9zkn6ecow")

# Function to connect to ClickHouse and get data
@st.cache_data
def get_data():
    # Calculate the date 45 days ago from today
    days_from = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
    days_to = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    # Query for calculated_main_telematics table with date filter
    query_main = f"""
    SELECT * FROM landing_zone_telematics.calculated_main_telematics 
    WHERE date >= '{days_from}'
    """
    result_main = client.query(query_main)
    df_main = pd.DataFrame(result_main.result_rows, columns=result_main.column_names)

    # Query for calculated_telematics_soc with start_date filter
    query_tel = f"""
    SELECT * FROM landing_zone_telematics.calculated_telematics_soc 
    WHERE start_date >= '{days_from}'
    """
    result_tel = client.query(query_tel)
    df_tel = pd.DataFrame(result_tel.result_rows, columns=result_tel.column_names)

    # Query for cohort data
    query_cohort = f"""
    SELECT 
        vehicle_number,
        reg_no,
        telematics_number,
        chassis_number,
        partner_id,
        deployed_city,
        product,
        date,
        total_discharge_soc,
        distance_bucket,
        total_discharge_soc_bucket,
        fast_charging_bucket,
        slow_charging_bucket,
        predicted_range_bucket,
        avg_distance,
        avg_c_d_cycles,
        avg_fast_charging,
        avg_slow_charging,
        avg_average_range
    FROM (
        SELECT 
            vehicle_number,
            reg_no,
            telematics_number,
            chassis_number,
            partner_id,
            deployed_city,
            product,
            date,
            SUM(total_discharge_soc) AS total_discharge_soc,
            AVG(total_km_travelled) AS avg_distance,
            AVG(total_discharge_soc) AS avg_c_d_cycles,
            AVG(fast_charge_soc) AS avg_fast_charging,
            AVG(slow_charge_soc) AS avg_slow_charging,
            AVG(predicted_range) AS avg_average_range,
            CASE 
                WHEN AVG(total_km_travelled) <= 10 THEN 'a. < 10 kms'
                WHEN AVG(total_km_travelled) > 10 AND AVG(total_km_travelled) <= 30 THEN 'b. 10 - 30 kms'
                WHEN AVG(total_km_travelled) > 30 AND AVG(total_km_travelled) <= 50 THEN 'c. 30 - 50 kms'
                WHEN AVG(total_km_travelled) > 50 AND AVG(total_km_travelled) <= 80 THEN 'd. 50 - 80 kms'
                WHEN AVG(total_km_travelled) > 80 AND AVG(total_km_travelled) <= 110 THEN 'e. 80 - 110 kms'
                WHEN AVG(total_km_travelled) > 110 AND AVG(total_km_travelled) <= 140 THEN 'f. 110 - 140 kms'
                ELSE 'g. > 140 kms'
            END AS distance_bucket,
            CASE 
                WHEN AVG(total_discharge_soc)*-1 <= 20 THEN '01. < 20'
                WHEN AVG(total_discharge_soc)*-1 > 20 AND AVG(total_discharge_soc)*-1 <= 50 THEN '02. 20 to 50'
                WHEN AVG(total_discharge_soc)*-1 > 50 AND AVG(total_discharge_soc)*-1 <= 80 THEN '03. 50 to 80'
                WHEN AVG(total_discharge_soc)*-1 > 80 AND AVG(total_discharge_soc)*-1 <= 120 THEN '04. 80 to 120'
                WHEN AVG(total_discharge_soc)*-1 > 120 AND AVG(total_discharge_soc)*-1 <= 170 THEN '05. 120 to 170'
                WHEN AVG(total_discharge_soc)*-1 > 170 AND AVG(total_discharge_soc)*-1 <= 240 THEN '06. 170 to 240'
                ELSE '07. > 240'
            END AS total_discharge_soc_bucket,
            CASE 
                WHEN AVG(fast_charge_soc) <= 20 THEN '01. < 0.20'
                WHEN AVG(fast_charge_soc) > 20 AND AVG(fast_charge_soc) <= 50 THEN '02. 20 to 50'
                WHEN AVG(fast_charge_soc) > 50 AND AVG(fast_charge_soc) <= 80 THEN '03. 50 to 80'
                WHEN AVG(fast_charge_soc) > 80 AND AVG(fast_charge_soc) <= 120 THEN '04. 80 to 120'
                WHEN AVG(fast_charge_soc) > 120 AND AVG(fast_charge_soc) <= 170 THEN '05. 120 to 170'
                WHEN AVG(fast_charge_soc) > 170 AND AVG(fast_charge_soc) <= 240 THEN '06. 170 to 240'
                ELSE '07. > 240'
            END AS fast_charging_bucket,
            CASE 
                WHEN AVG(slow_charge_soc) <= 20 THEN '01. < 0.20'
                WHEN AVG(slow_charge_soc) > 20 AND AVG(slow_charge_soc) <= 50 THEN '02. 20 to 50'
                WHEN AVG(slow_charge_soc) > 50 AND AVG(slow_charge_soc) <= 80 THEN '03. 50 to 80'
                WHEN AVG(slow_charge_soc) > 80 AND AVG(slow_charge_soc) <= 120 THEN '04. 80 to 120'
                WHEN AVG(slow_charge_soc) > 120 AND AVG(slow_charge_soc) <= 170 THEN '05. 120 to 170'
                WHEN AVG(slow_charge_soc) > 170 AND AVG(slow_charge_soc) <= 240 THEN '06. 170 to 240'
                ELSE '07. > 240'
            END AS slow_charging_bucket,
            CASE 
                WHEN AVG(predicted_range) <= 40 THEN 'a. <= 40 kms'
                WHEN AVG(predicted_range) > 40 AND AVG(predicted_range) <= 50 THEN 'b. 40 - 50 kms'
                WHEN AVG(predicted_range) > 50 AND AVG(predicted_range) <= 60 THEN 'c. 50 - 60 kms'
                WHEN AVG(predicted_range) > 60 AND AVG(predicted_range) <= 70 THEN 'd. 60 - 70 kms'
                WHEN AVG(predicted_range) > 70 AND AVG(predicted_range) <= 80 THEN 'e. 70 - 80 kms'
                WHEN AVG(predicted_range) > 80 AND AVG(predicted_range) <= 90 THEN 'f. 80 - 90 kms'
                WHEN AVG(predicted_range) > 90 AND AVG(predicted_range) <= 100 THEN 'g. 90 - 100 kms'
                ELSE 'h. > 100 kms'
            END AS predicted_range_bucket
        FROM landing_zone_telematics.calculated_main_telematics
        WHERE date >= '{days_from}'
        GROUP BY vehicle_number, reg_no, telematics_number, chassis_number, partner_id, deployed_city, product, date
    )
    """
    result_cohort = client.query(query_cohort)
    df_cohort = pd.DataFrame(result_cohort.result_rows, columns=result_cohort.column_names)

    return df_main.copy(), df_tel.copy(), df_cohort.copy()
    
@st.cache_data
def get_mapping_data():
    conn = psycopg2.connect(
        database="postgres",
        user='postgres.gqmpfexjoachyjgzkhdf',
        password='Change@2015Log9',
        host='aws-0-ap-south-1.pooler.supabase.com',
        port='5432'
    )

    query_mapping = "SELECT reg_no, chassis_number, telematics_number, location, client_name, battery_type FROM mapping_table;"
    df_mapping = pd.read_sql_query(query_mapping, conn)
    conn.close()
    
    return df_mapping.copy()

def replace_invalid_values(series, placeholder, invalid_values):
    return series.replace(invalid_values, placeholder).fillna(placeholder)

    
def main():
    # if 'last_refresh' not in st.session_state:
    #     st.session_state['last_refresh'] = time.time()
    # # Check if the refresh interval has passed
    # refresh_interval = 3600  # seconds (e.g., 300 seconds = 5 minutes)
    # current_time = time.time()
    # if current_time - st.session_state['last_refresh'] > refresh_interval:
    #     st.session_state['last_refresh'] = current_time
    #     st.cache_data.clear()
    #     st.experimental_rerun()
        
    df_main, df_tel,df_cohort = get_data()
    df_mapping = get_mapping_data()
    df = df_main  # Use df for data from calculated_main_telematics table
    df2 = df_tel  # Use df2 for data from calculated_telematics_soc
    df3 = df_cohort  # Use df3 for cohorting data
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        df_main['date'] = pd.to_datetime(df_main['date'], errors='coerce')
        
        # Calculate the maximum and minimum dates in the dataset
        max_date = df_main['date'].max()
        min_date = df_main['date'].min()
        
        # Calculate the start date for the last 7 days range
        start_date_last_7_days = max_date - pd.Timedelta(days=6)
        
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
        df_filtered_cohort = df_cohort
        df_filtered_mapping = df_mapping
        
        if len(date_range) == 2 and date_range[0] and date_range[1]:
                # ... [your existing data filtering code]
                df_filtered = df_filtered[(df_filtered['date'] >= pd.Timestamp(date_range[0])) & (df_filtered['date'] <= pd.Timestamp(date_range[1]))]
                
                df_filtered_tel['end_date'] = pd.to_datetime(df_filtered_tel['end_date'])
                
                df_filtered_cohort['date'] = pd.to_datetime(df_filtered_cohort['date'])
            
                df_filtered_tel = df_filtered_tel[(df_filtered_tel['end_date'] >= pd.Timestamp(date_range[0])) & (df_filtered_tel['end_date'] <= pd.Timestamp(date_range[1]))]

                df_filtered_cohort = df_filtered_cohort[(df_filtered_cohort['date'] >= pd.Timestamp(date_range[0])) & (df_filtered_cohort['date'] <= pd.Timestamp(date_range[1]))]
            
                # Customer Name filter
                partner_ids = df_filtered['partner_id'].dropna().unique().tolist()
                selected_partner_ids = st.multiselect('Customer Name', partner_ids)
    
                # Filter dataframe by customer name if selected
                if selected_partner_ids:
                    df_filtered = df_filtered[df_filtered['partner_id'].isin(selected_partner_ids)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['partner_id'].isin(selected_partner_ids)]
                    df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['partner_id'].isin(selected_partner_ids)]
                    df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['client_name'].isin(selected_partner_ids)]

    
                # Product filter
                products = df_filtered['product'].dropna().unique().tolist()
                selected_products = st.multiselect('Product', products)
    
                # Filter dataframe by product if selected
                if selected_products:
                    df_filtered = df_filtered[df_filtered['product'].isin(selected_products)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['product'].isin(selected_products)]
                    df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['product'].isin(selected_products)]
                    df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['battery_type'].isin(selected_products)]

    
                # Registration Number filter
                reg_nos = df_filtered['reg_no'].dropna().unique().tolist()
                selected_reg_nos = st.multiselect('Registration Number', reg_nos)
    
                # Filter dataframe by registration number if selected
                if selected_reg_nos:
                    df_filtered = df_filtered[df_filtered['reg_no'].isin(selected_reg_nos)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['reg_no'].isin(selected_reg_nos)]
                    df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['reg_no'].isin(selected_reg_nos)]
                    df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['reg_no'].isin(selected_reg_nos)]

                
                # Chassis Number filter
                chassis_nos = df_filtered['chassis_number'].dropna().unique().tolist()
                selected_chassis_nos = st.multiselect('Chassis Number', chassis_nos)
    
                # Filter dataframe by registration number if selected
                if selected_chassis_nos:
                    df_filtered = df_filtered[df_filtered['chassis_number'].isin(selected_chassis_nos)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['chassis_number'].isin(selected_chassis_nos)]
                    df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['chassis_number'].isin(selected_chassis_nos)]
                    df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['chassis_number'].isin(selected_chassis_nos)]


                # City filter
                cities = df_filtered['deployed_city'].dropna().unique().tolist()
                selected_cities = st.multiselect('City', cities)
    
                # Filter dataframe by city if selected
                if selected_cities:
                    df_filtered = df_filtered[df_filtered['deployed_city'].isin(selected_cities)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['deployed_city'].isin(selected_cities)]
                    df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['deployed_city'].isin(selected_cities)]
                    df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['location'].isin(selected_cities)]
                
                # Update the duration slider based on filtered data
                if not df_filtered.empty:
                    min_km, max_km = df_filtered['total_km_travelled'].agg(['min', 'max'])
                    min_km, max_km = int(min_km), int(max_km)
                else:
                    min_km, max_km = 0, 0  # Defaults when no data is available
                
                # Set the initial value of the slider to start at 10, or at the minimum value if it's higher than 10
                initial_min_km = max(10, min_km)
                
                km_range = st.slider("Select daily distance travelled range(kms)", min_km, max_km, (initial_min_km, max_km))
                
                # Apply the duration filter
                df_filtered = df_filtered[(df_filtered['total_km_travelled'] >= km_range[0]) & (df_filtered['total_km_travelled'] <= km_range[1])]

        st.markdown("### Cache Management")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.experimental_rerun()

    
    
    # Layout: 3 columns
    col1, col2, col3 = st.columns(3)

    # Column 1 - Average Distance Travelled
    with col1:
        st.markdown("## Average Distance")

        # Box plot data preparation
        if not df_filtered.empty:
            
            grouped_distance = df_filtered.groupby('date')['total_km_travelled']
            
            avg_dist_all_vehicles_per_day = df_filtered.groupby('date')['total_km_travelled'].mean().round(1).reset_index()
            overall_avg_dist_per_day = avg_dist_all_vehicles_per_day['total_km_travelled'].mean().round(1)
            
            st.metric(" ", f"{overall_avg_dist_per_day:.2f} km")
            
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
                "series": [{"name": "Distance Travelled", "type": "boxplot", "data": []}]
            }
    
            for name, group in grouped_distance:
                percentiles = group.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist()
                boxplot_data["xAxis"]["data"].append(name.strftime('%d-%m-%Y'))
                boxplot_data["series"][0]["data"].append([
                    percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
                ])
    
            st_echarts(options=boxplot_data, height="400px")
        else:
            st.write("Please select a date range and other filters to view analytics.")
        
       
        
        # if 'df_filtered' in locals() and not df_filtered.empty:
            
        #     avg_dist_all_vehicles_per_day = df_filtered.groupby('date')['total_km_travelled'].mean().round(1).reset_index()
        #     overall_avg_dist_per_day = avg_dist_all_vehicles_per_day['total_km_travelled'].mean().round(1)
        #     st.metric(" ", f"{overall_avg_dist_per_day:.2f} km")

        #     # Plotting the average distance travelled for all vehicles per day
        #     options = {
        #         "xAxis": {
        #             "type": "category",
        #             "data": avg_dist_all_vehicles_per_day['date'].dt.strftime('%d-%m-%Y').tolist(),
        #             "axisLabel": {
        #                 "rotate": 90,  # Rotating x-axis labels by 90 degrees
        #                 "fontSize": 10  # Reducing font size
        #             }
        #         },
        #         "yAxis": {
        #             "type": "value"
        #         },
        #         "tooltip": {
        #             "trigger": "axis",
        #             "axisPointer": {
        #                 "type": "cross"
        #             }
        #         },
        #         "series": [{
        #             "data": avg_dist_all_vehicles_per_day['total_km_travelled'].tolist(),
        #             "type": "line"
        #         }]
        #     }
        #     st_echarts(options=options, height="400px")


        # else:
        #     st.write("Please select a date range and other filters to view analytics.")
        
        st.markdown("## Charge SOC Over Time")
        if 'df_filtered' in locals() and not df_filtered.empty:
            # Group by date and calculate averages
            charge_soc_data = df_filtered.groupby('date').agg({
                'fast_charge_soc': 'mean',
                'slow_charge_soc': 'mean'
            }).round(1).reset_index()

            # Prepare data for the stacked column chart
            stacked_column_options = {
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {
                        "type": "shadow"
                    }
                },
                "xAxis": {
                    "type": "category",
                    "rotate": 90,
                    "data": charge_soc_data['date'].dt.strftime('%d-%m-%Y').tolist()
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "name": "Fast Charge SOC",
                        "type": "bar",
                        "stack": "charging",
                        "data": charge_soc_data['fast_charge_soc'].tolist()
                    },
                    {
                        "name": "Slow Charge SOC",
                        "type": "bar",
                        "stack": "charging",
                        "data": charge_soc_data['slow_charge_soc'].tolist()
                    }
                ]
            }
            st_echarts(options=stacked_column_options, height="400px")
        else:
            st.write("Please select a date range and other filters to view analytics.") 
    # Column 2 - Average Range of the Fleet
    with col2:
        st.markdown("## Average Range")
    
        # Filter df_filtered for total_km_travelled > 15km
        df_range = df_filtered[(df_filtered['total_km_travelled'] >= 0) & (df_filtered['total_discharge_soc'] < 0)]

        # Average Range of the Fleet Metric Calculation
        avg_range_fleet = np.sum(df_range['total_km_travelled']) * -100 / np.sum(df_range['total_discharge_soc'])
        st.metric(" ", f"{avg_range_fleet:.2f} km")
    
        # Box plot data preparation
        if not df_range.empty:
            grouped = df_range.groupby('date')['predicted_range']
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
                "series": [{"name": "Predicted Range", "type": "boxplot", "data": []}]
            }
    
            for name, group in grouped:
                percentiles = group.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist()
                boxplot_data["xAxis"]["data"].append(name.strftime('%d-%m-%Y'))
                boxplot_data["series"][0]["data"].append([
                    percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
                ])
    
            st_echarts(options=boxplot_data, height="400px")
        else:
            st.write("Please select a date range and other filters to view analytics.")

        st.markdown("## Charging Metrics")
        if 'df_filtered' in locals() and not df_filtered.empty:
            avg_slow_charge_sessions = df_filtered['slow_charge_count'].mean()
            avg_slow_charge_soc = df_filtered['slow_charge_soc'].mean()
            avg_fast_charge_sessions = df_filtered['fast_charge_count'].mean()
            avg_fast_charge_soc = df_filtered['fast_charge_soc'].mean()
            
            st.metric("Average Slow Charge Sessions in a day", f"{avg_slow_charge_sessions:.2f}")
            st.metric("Average Slow Charge SOC in a day", f"{avg_slow_charge_soc:.2f}")
            st.metric("Average Fast Charge Sessions in a day", f"{avg_fast_charge_sessions:.2f}")
            st.metric("Average Fast Charge SOC in a day", f"{avg_fast_charge_soc:.2f}")
        else:
            st.write("Please select a date range and other filters to view analytics.")
    
    with col3:
        st.markdown("## Distance travelled, Range and Runtime")
            
        # Filter df_filtered for total_km_travelled >= 0 and total_discharge_soc < 0
        df_range = df_filtered[(df_filtered['total_km_travelled'] >= 0)]
            
        if not df_range.empty:
            # Define invalid values for each column
            invalid_reg_no_values = [None, np.nan, "NA", "0", "FALSE", "NULL","false","False"]
            invalid_telematics_values = [None, np.nan, "111111111111111", "FALSE", "11111111111111", "A","false","False"]
            
            # Fill missing or invalid values with placeholders
            df_range['chassis_number'] = replace_invalid_values(df_range['chassis_number'], 'Unknown Chassis', [None, np.nan, False, 0, '0'])
            df_range['reg_no'] = replace_invalid_values(df_range['reg_no'], 'Unknown Reg', invalid_reg_no_values)
            df_range['telematics_number'] = replace_invalid_values(df_range['telematics_number'], 'Unknown Telematics', invalid_telematics_values)
            
            # Group by reg_no, chassis_number, and telematics_number, and calculate the sum of total_km_travelled, total_discharge_soc, and total_runtime_minutes
            df_grouped = df_range.groupby(['chassis_number', 'reg_no', 'telematics_number'], dropna=False).agg(
                total_km_travelled_sum=('total_km_travelled', 'sum'),
                total_discharge_soc_sum=('total_discharge_soc', 'sum'),
                total_runtime_minutes_sum=('total_runtime_minutes', 'sum')
            ).reset_index()
            
            # Calculate Range
            df_grouped['Range'] = df_grouped['total_km_travelled_sum'] * (-100) / df_grouped['total_discharge_soc_sum']
            
            # Calculate Average Run Time
            avg_run_time = df_range['total_runtime_minutes'].median()
            st.markdown(f"#### Average Run Time: {avg_run_time:.2f} minutes")
            
            # Calculate Average Run Time Per Day for each group
            avg_run_time_per_day = df_range.groupby(['chassis_number', 'reg_no', 'telematics_number'], dropna=False)['total_runtime_minutes'].median().reset_index(name='avg_run_time_per_day')
            df_grouped = df_grouped.merge(avg_run_time_per_day, on=['chassis_number', 'reg_no', 'telematics_number'])
            
            # Replace placeholders with actual missing value representations
            df_grouped['chassis_number'].replace('Unknown Chassis', None, inplace=True)
            df_grouped['reg_no'].replace('Unknown Reg', None, inplace=True)
            df_grouped['telematics_number'].replace('Unknown Telematics', None, inplace=True)
            
            # Format the DataFrame to match the screenshot layout
            df_display = df_grouped[['chassis_number', 'reg_no', 'telematics_number', 'total_km_travelled_sum', 'total_runtime_minutes_sum', 'avg_run_time_per_day', 'Range']].rename(
                columns={
                    'chassis_number': 'Chassis Number',
                    'reg_no': 'Registration Number', 
                    'telematics_number': 'Telematics Number',
                    'total_km_travelled_sum': 'Total KM Travelled',
                    'total_runtime_minutes_sum': 'Total Runtime Minutes',
                    'avg_run_time_per_day': 'Average Run Time Per Day'
                }
            )
            
            st.dataframe(df_display, height=400)
        # if not df_range.empty:
        #     # Group by reg_no and calculate the sum of total_km_travelled and the Range
        #     df_grouped = df_range.groupby('reg_no').agg(
        #         total_km_travelled_sum=('total_km_travelled', 'sum'),
        #         total_discharge_soc_sum=('total_discharge_soc', 'sum')
        #     ).reset_index()
        #     df_grouped['Range'] = df_grouped['total_km_travelled_sum'] * (-100) / df_grouped['total_discharge_soc_sum']

        #     # Format the DataFrame to match the screenshot layout
        #     df_display = df_grouped[['reg_no', 'total_km_travelled_sum', 'Range']].rename(
        #         columns={'reg_no': 'Registration Number', 'total_km_travelled_sum': 'Total KM Travelled'}
        #     )
            

            
        #     st.dataframe(df_display,height=400)
            
        st.markdown("## Vehicle level Average Slow and Fast Charge SOC")

        # Calculate the average slow_charge_soc and average fast_charge_soc grouped by reg_no
        df_charging = df_filtered.groupby('reg_no').agg(
            average_slow_charge_soc=('slow_charge_soc', 'mean'),
            average_fast_charge_soc=('fast_charge_soc', 'mean')
        ).reset_index().round(1)  # Round to 1 decimal place

        # Rename columns for display
        df_charging.rename(columns={
            'reg_no': 'Registration Number',
            'average_slow_charge_soc': 'Average Slow Charge SOC',
            'average_fast_charge_soc': 'Average Fast Charge SOC'
        }, inplace=True)

        # Display the DataFrame
        st.dataframe(df_charging, height=300)     
        
    df_charging_locations = df_filtered_tel[(df_filtered_tel['change_in_soc'] > 0) & (df_filtered_tel['soc_type'] == "Charging")]


    # if not df_charging_locations.empty:
    #     # Check if the necessary columns exist
    #     required_columns = ['charging_location', 'charging_location_coordinates', 'change_in_soc', 'soc_type']
    #     missing_columns = [col for col in required_columns if col not in df_charging_locations.columns]

    #     if missing_columns:
    #         st.write(f"Missing columns in df_charging_locations: {', '.join(missing_columns)}")
    #         return

    #     # Create a DataFrame with charging_location, charging_location_coordinates, and count of Charging soc_type
    #     df_map_data = df_charging_locations.groupby(['charging_location', 'charging_location_coordinates']).agg({
    #         'change_in_soc': 'sum',  # Calculate the sum of total_charge_soc
    #         'soc_type': 'size'  # Count the number of soc_type
    #     }).reset_index()

    #     # Extract latitude and longitude from charging_location_coordinates
    #     df_map_data[['latitude', 'longitude']] = df_map_data['charging_location_coordinates'].str.split(',', expand=True).astype(float)

    #     # Define the color scale for the bars based on total_charge_soc
    #     color_range = [0, df_map_data['change_in_soc'].max()]  # Adjust as needed

    #     # Create a custom hover text for data points
    #     df_map_data['hover_text'] = df_map_data.apply(
    #         lambda row: f"Location: {row['charging_location']}<br>"
    #                     f"Total Charge SOC: {row['change_in_soc']}<br>"
    #                     f"SOC Type Count: {row['soc_type']}",
    #         axis=1
    #     )

    #     # Create a scatter map using Plotly Express with Mapbox
    #     fig = px.scatter_mapbox(
    #         df_map_data,
    #         lat="latitude",
    #         lon="longitude",
    #         color="change_in_soc",
    #         size="soc_type",
    #         color_continuous_scale="YlOrRd",  # You can choose a different color scale
    #         size_max=15,  # Adjust the max size of data points
    #         zoom=10,  # Adjust the initial zoom level
    #         hover_name="charging_location",
    #         hover_data=["change_in_soc", "soc_type", "hover_text"],
    #     )

    #     # Customize the map layout
    #     fig.update_layout(
    #         mapbox_style="streets",  # You can choose different Mapbox styles
    #         margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove margin
    #         template="plotly_dark"
    #     )

    #     # Display the map
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     st.write("No charging locations found.")

    


    if not df_filtered_cohort.empty:                
        st.markdown("## Distance Travelled distribution ")
        # Create a pivot table
        pivot_table = df_filtered_cohort.pivot_table(
            index='distance_bucket',  # Rows (index) will be the different distance buckets
            columns='date',           # Columns will be the dates
            values='vehicle_number',  # Values will be the count of vehicle numbers
            aggfunc='count',          # We count the number of vehicle numbers for each (row, column) pair
            fill_value=0              # Fill missing values with 0
        )
        pivot_table.columns = pivot_table.columns.strftime('%d-%m-%Y')
        
        # Convert the cell values to percentage of row totals
        pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=0), axis='columns') * 100
        
        # Format the numbers in the pivot table to show as percentages with 1 decimal place and append "%"
        formatted_pivot_table_percentage = pivot_table_percentage.style.format("{:.1f}%")
        
        # Now, display the formatted pivot table in Streamlit
        st.table(formatted_pivot_table_percentage)

    else:
        st.write("Please select a date range and other filters to view analytics.")
        

    # else:
    #     st.write("Please select a date range and other filters to view analytics.")
    
    if not df_filtered.empty:
        st.markdown("## Day Wise Data")
        st.dataframe(df_filtered, height=300)
        
        # # PyGWalker integration for df_filtered
        # st.markdown("## Explore Day Wise Data")
        # pyg.walk(df_filtered, "Day Wise Data Exploration")
    else:
        st.write("df_filtered is empty")
        
    if not df_filtered_tel.empty:
        st.markdown("## SOC Data")
        st.dataframe(df_filtered_tel, height=300)
        
        # # PyGWalker integration for df_filtered_tel
        # st.markdown("## Explore SOC Data")
        # pyg.walk(df_filtered_tel, "SOC Data Exploration")
    else:
        st.write("df_filtered_tel is empty")

    # Define invalid values for each column
    invalid_reg_no_values = [None, np.nan, "NA", "0", "FALSE", "NULL"]
    invalid_telematics_values = [None, np.nan, "111111111111111", "FALSE", "11111111111111", "A"]

    # Fill missing or invalid values with placeholders
    df_filtered_mapping['chassis_number'] = replace_invalid_values(df_filtered_mapping['chassis_number'], 'Unknown Chassis', [None, np.nan, False, 0, '0'])
    df_filtered_mapping['reg_no'] = replace_invalid_values(df_filtered_mapping['reg_no'], 'Unknown Reg', invalid_reg_no_values)
    df_filtered_mapping['telematics_number'] = replace_invalid_values(df_filtered_mapping['telematics_number'], 'Unknown Telematics', invalid_telematics_values)

    # Add row numbers
    df_filtered_mapping.reset_index(drop=True, inplace=True)
    df_filtered_mapping['Row Number'] = df_filtered_mapping.index + 1

    # Display the "List of Assets" DataFrame
    if not df_filtered_mapping.empty:
        st.markdown("## List of Assets")
        st.dataframe(df_filtered_mapping[['Row Number', 'chassis_number', 'reg_no', 'telematics_number', 'location', 'client_name', 'battery_type']], height=300)
    else:
        st.write("No assets found for the selected filters.")

    
    # Display the filtered dataframe below the charts
    # if not df_filtered_tel.empty:
    #     st.markdown("## Cohorting Data")
    #     st.dataframe(df_filtered_cohort, height=300)

    # df_filtered_tel['energy_consumption'] = df_filtered_tel.apply(
    #     lambda row: round((11.77 * 1000 * (row['change_in_soc'] / -100)) / row['total_distance_km'], 2) 
    #     if (row['product'] == "12_KW_4W" and row['soc_type'] == "Discharging" and row['total_distance_km'] != 0)
    #     else row['energy_consumption'],
    #     axis=1
    # )
    
    # # Remove the "primary_id" column from df_filtered_tel
    # df_filtered_tel_without_primary_id = df_filtered_tel.drop(columns=['primary_id'])

    # # Display the "Day Wise Summary" DataFrame without the "primary_id" column
    # st.markdown("## Day Wise Summary")
    # st.dataframe(df_filtered_tel_without_primary_id, height=300)
        


if __name__ == "__main__":
    main()
