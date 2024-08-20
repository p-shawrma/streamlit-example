import streamlit as st
import psycopg2
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
px.set_mapbox_access_token("pk.eyJ1IjoicC1zaGFybWEiLCJhIjoiY2xzNjRzbTY1MXNodjJsbXUwcG0wNG50ciJ9.v32bwq-wi6whz9zkn6ecow")

# Function to connect to ClickHouse and get data
@st.cache_data
def get_data():
    # Calculate the date 45 days ago from today
    days_from = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
    days_to = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')

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

    # Cohort Data
    cohort_data = df_main.groupby([
        'vehicle_number', 'reg_no', 'telematics_number', 'chassis_number', 
        'partner_id', 'deployed_city', 'product', 'date'
    ]).agg(
        total_discharge_soc=('total_discharge_soc', 'sum'),
        avg_distance=('total_km_travelled', 'mean'),
        avg_c_d_cycles=('total_discharge_soc', 'mean'),
        avg_fast_charging=('fast_charge_soc', 'mean'),
        avg_slow_charging=('slow_charge_soc', 'mean'),
        avg_average_range=('predicted_range', 'mean')
    ).reset_index()

    # Adding buckets
    cohort_data['distance_bucket'] = pd.cut(
        cohort_data['avg_distance'],
        bins=[-float('inf'), 10, 30, 50, 80, 110, 140, float('inf')],
        labels=['a. < 10 kms', 'b. 10 - 30 kms', 'c. 30 - 50 kms', 'd. 50 - 80 kms', 
                'e. 80 - 110 kms', 'f. 110 - 140 kms', 'g. > 140 kms']
    )

    cohort_data['total_discharge_soc_bucket'] = pd.cut(
        cohort_data['avg_c_d_cycles'] * -1,
        bins=[-float('inf'), 20, 50, 80, 120, 170, 240, float('inf')],
        labels=['01. < 20', '02. 20 to 50', '03. 50 to 80', '04. 80 to 120', 
                '05. 120 to 170', '06. 170 to 240', '07. > 240']
    )

    cohort_data['fast_charging_bucket'] = pd.cut(
        cohort_data['avg_fast_charging'],
        bins=[-float('inf'), 20, 50, 80, 120, 170, 240, float('inf')],
        labels=['01. < 0.20', '02. 20 to 50', '03. 50 to 80', '04. 80 to 120', 
                '05. 120 to 170', '06. 170 to 240', '07. > 240']
    )

    cohort_data['slow_charging_bucket'] = pd.cut(
        cohort_data['avg_slow_charging'],
        bins=[-float('inf'), 20, 50, 80, 120, 170, 240, float('inf')],
        labels=['01. < 0.20', '02. 20 to 50', '03. 50 to 80', '04. 80 to 120', 
                '05. 120 to 170', '06. 170 to 240', '07. > 240']
    )

    cohort_data['predicted_range_bucket'] = pd.cut(
        cohort_data['avg_average_range'],
        bins=[-float('inf'), 40, 50, 60, 70, 80, 90, 100, float('inf')],
        labels=['a. <= 40 kms', 'b. 40 - 50 kms', 'c. 50 - 60 kms', 'd. 60 - 70 kms', 
                'e. 70 - 80 kms', 'f. 80 - 90 kms', 'g. 90 - 100 kms', 'h. > 100 kms']
    )

    return df_main.copy(), df_tel.copy(), cohort_data.copy()
    
@st.cache_data
def get_mapping_data():
    client = clickhouse_connect.get_client(
        host=ch_host,
        user=ch_user,
        password=ch_password,
        database=ch_database,
        secure=True
    )
    
    query_mapping = "SELECT reg_no, chassis_number, telematics_number, location, client_name, battery_type FROM mapping_table"
    result_mapping = client.query(query_mapping)
    df_mapping = pd.DataFrame(result_mapping.result_rows, columns=result_mapping.column_names)
    
    return df_mapping.copy()


def replace_invalid_values(series, placeholder, invalid_values):
    return series.replace(invalid_values, placeholder).fillna(placeholder)

def main():
    df_main, df_tel, df_cohort = get_data()
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
            df_filtered = df_filtered[(df_filtered['date'] >= pd.Timestamp(date_range[0])) & (df_filtered['date'] <= pd.Timestamp(date_range[1]))]
            df_filtered_tel['end_date'] = pd.to_datetime(df_filtered_tel['end_date'])
            df_filtered_cohort['date'] = pd.to_datetime(df_filtered_cohort['date'])
            df_filtered_tel = df_filtered_tel[(df_filtered_tel['end_date'] >= pd.Timestamp(date_range[0])) & (df_filtered_tel['end_date'] <= pd.Timestamp(date_range[1]))]
            df_filtered_cohort = df_filtered_cohort[(df_filtered_cohort['date'] >= pd.Timestamp(date_range[0])) & (df_filtered_cohort['date'] <= pd.Timestamp(date_range[1]))]
            
            # Customer Name filter
            partner_ids = df_filtered_mapping['client_name'].dropna().unique().tolist()
            selected_partner_ids = st.multiselect('Customer Name', partner_ids)

            if selected_partner_ids:
                df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['client_name'].isin(selected_partner_ids)]
                filtered_telematics_numbers = df_filtered_mapping['telematics_number'].unique().tolist()
                df_filtered = df_filtered[df_filtered['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_tel = df_filtered_tel[df_filtered_tel['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['telematics_number'].isin(filtered_telematics_numbers)]

            # Product filter
            products = df_filtered_mapping['battery_type'].dropna().unique().tolist()
            selected_products = st.multiselect('Product', products)

            if selected_products:
                df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['battery_type'].isin(selected_products)]
                filtered_telematics_numbers = df_filtered_mapping['telematics_number'].unique().tolist()
                df_filtered = df_filtered[df_filtered['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_tel = df_filtered_tel[df_filtered_tel['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['telematics_number'].isin(filtered_telematics_numbers)]

            # Registration Number filter
            reg_nos = df_filtered_mapping['reg_no'].dropna().unique().tolist()
            selected_reg_nos = st.multiselect('Registration Number', reg_nos)

            if selected_reg_nos:
                df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['reg_no'].isin(selected_reg_nos)]
                filtered_telematics_numbers = df_filtered_mapping['telematics_number'].unique().tolist()
                df_filtered = df_filtered[df_filtered['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_tel = df_filtered_tel[df_filtered_tel['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['telematics_number'].isin(filtered_telematics_numbers)]

            # Chassis Number filter
            chassis_nos = df_filtered_mapping['chassis_number'].dropna().unique().tolist()
            selected_chassis_nos = st.multiselect('Chassis Number', chassis_nos)

            if selected_chassis_nos:
                df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['chassis_number'].isin(selected_chassis_nos)]
                filtered_telematics_numbers = df_filtered_mapping['telematics_number'].unique().tolist()
                df_filtered = df_filtered[df_filtered['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_tel = df_filtered_tel[df_filtered_tel['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['telematics_number'].isin(filtered_telematics_numbers)]

            # City filter
            cities = df_filtered_mapping['location'].dropna().unique().tolist()
            selected_cities = st.multiselect('City', cities)

            if selected_cities:
                df_filtered_mapping = df_filtered_mapping[df_filtered_mapping['location'].isin(selected_cities)]
                filtered_telematics_numbers = df_filtered_mapping['telematics_number'].unique().tolist()
                df_filtered = df_filtered[df_filtered['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_tel = df_filtered_tel[df_filtered_tel['telematics_number'].isin(filtered_telematics_numbers)]
                df_filtered_cohort = df_filtered_cohort[df_filtered_cohort['telematics_number'].isin(filtered_telematics_numbers)]

            # Update the duration slider based on filtered data
            if not df_filtered.empty:
                min_km, max_km = df_filtered['total_km_travelled'].agg(['min', 'max'])
                min_km, max_km = int(min_km), int(max_km)
            else:
                min_km, max_km = 0, 0  # Defaults when no data is available

            initial_min_km = max(10, min_km)
            km_range = st.slider("Select daily distance travelled range(kms)", min_km, max_km, (initial_min_km, max_km))
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
                        "rotate": 90,
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

        st.markdown("## Charge SOC Over Time")
        if 'df_filtered' in locals() and not df_filtered.empty:
            charge_soc_data = df_filtered.groupby('date').agg({
                'fast_charge_soc': 'mean',
                'slow_charge_soc': 'mean'
            }).round(1).reset_index()

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
    
        df_range = df_filtered[(df_filtered['total_km_travelled'] >= 0) & (df_filtered['total_discharge_soc'] < 0)]
        avg_range_fleet = np.sum(df_range['total_km_travelled']) * -100 / np.sum(df_range['total_discharge_soc'])
        st.metric(" ", f"{avg_range_fleet:.2f} km")
    
        if not df_range.empty:
            grouped = df_range.groupby('date')['predicted_range']
            boxplot_data = {
                "xAxis": {
                    "type": "category",
                    "data": [],
                    "axisLabel": {
                        "rotate": 90,
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
            
        df_range = df_filtered[(df_filtered['total_km_travelled'] >= 0)]
            
        if not df_range.empty:
            invalid_reg_no_values = [None, np.nan, "NA", "0", "FALSE", "NULL","false","False"]
            invalid_telematics_values = [None, np.nan, "111111111111111", "FALSE", "11111111111111", "A","false","False"]
            
            df_range['chassis_number'] = replace_invalid_values(df_range['chassis_number'], 'Unknown Chassis', [None, np.nan, False, 0, '0'])
            df_range['reg_no'] = replace_invalid_values(df_range['reg_no'], 'Unknown Reg', invalid_reg_no_values)
            df_range['telematics_number'] = replace_invalid_values(df_range['telematics_number'], 'Unknown Telematics', invalid_telematics_values)
            
            df_grouped = df_range.groupby(['chassis_number', 'reg_no', 'telematics_number'], dropna=False).agg(
                total_km_travelled_sum=('total_km_travelled', 'sum'),
                total_discharge_soc_sum=('total_discharge_soc', 'sum'),
                total_runtime_minutes_sum=('total_runtime_minutes', 'sum')
            ).reset_index()
            
            df_grouped['Range'] = df_grouped['total_km_travelled_sum'] * (-100) / df_grouped['total_discharge_soc_sum']
            
            avg_run_time = df_range['total_runtime_minutes'].median()
            st.markdown(f"#### Average Run Time: {avg_run_time:.2f} minutes")
            
            avg_run_time_per_day = df_range.groupby(['chassis_number', 'reg_no', 'telematics_number'], dropna=False)['total_runtime_minutes'].median().reset_index(name='avg_run_time_per_day')
            df_grouped = df_grouped.merge(avg_run_time_per_day, on=['chassis_number', 'reg_no', 'telematics_number'])
            
            df_grouped['chassis_number'].replace('Unknown Chassis', None, inplace=True)
            df_grouped['reg_no'].replace('Unknown Reg', None, inplace=True)
            df_grouped['telematics_number'].replace('Unknown Telematics', None, inplace=True)
            
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
        
        st.markdown("## Vehicle level Average Slow and Fast Charge SOC")
        
        df_charging_1 = df_filtered[(df_filtered['slow_charge_soc'] >= 0) | df_filtered['fast_charge_soc'] >= 0]
        
        # Replace invalid values
        df_charging_1['chassis_number'] = replace_invalid_values(df_charging_1['chassis_number'], 'Unknown Chassis', [None, np.nan, False, 0, '0'])
        df_charging_1['reg_no'] = replace_invalid_values(df_charging_1['reg_no'], 'Unknown Reg', invalid_reg_no_values)
        df_charging_1['telematics_number'] = replace_invalid_values(df_charging_1['telematics_number'], 'Unknown Telematics', invalid_telematics_values)
        
        # Grouping by vehicle details and calculating mean and total SOC values
        df_charging_grouped = df_charging_1.groupby(['chassis_number', 'reg_no', 'telematics_number'], dropna=False).agg(
            total_slow_charge_soc=('slow_charge_soc', 'sum'),
            total_fast_charge_soc=('fast_charge_soc', 'sum'),
            average_slow_charge_soc=('slow_charge_soc', 'mean'),
            average_fast_charge_soc=('fast_charge_soc', 'mean')
        ).reset_index()
        
        # Replace placeholders back to None for display
        df_charging_grouped['chassis_number'].replace('Unknown Chassis', None, inplace=True)
        df_charging_grouped['reg_no'].replace('Unknown Reg', None, inplace=True)
        df_charging_grouped['telematics_number'].replace('Unknown Telematics', None, inplace=True)
        
        df_display_charging = df_charging_grouped[['chassis_number', 'reg_no', 'telematics_number', 'total_slow_charge_soc', 'total_fast_charge_soc', 'average_slow_charge_soc', 'average_fast_charge_soc']].rename(
            columns={
                'chassis_number': 'Chassis Number',
                'reg_no': 'Registration Number',
                'telematics_number': 'Telematics Number',
                'total_slow_charge_soc': 'Total Slow Charge SOC',
                'total_fast_charge_soc': 'Total Fast Charge SOC',
                'average_slow_charge_soc': 'Average Slow Charge SOC',
                'average_fast_charge_soc': 'Average Fast Charge SOC'
            }
        )
        
        st.dataframe(df_display_charging, height=300) 
        
    df_charging_locations = df_filtered_tel[(df_filtered_tel['change_in_soc'] > 0) & (df_filtered_tel['soc_type'] == "Charging")]

    if not df_filtered_cohort.empty:                
        st.markdown("## Distance Travelled distribution ")
        pivot_table = df_filtered_cohort.pivot_table(
            index='distance_bucket',
            columns='date',
            values='vehicle_number',
            aggfunc='count',
            fill_value=0
        )
        pivot_table.columns = pivot_table.columns.strftime('%d-%m-%Y')
        
        pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=0), axis='columns') * 100
        formatted_pivot_table_percentage = pivot_table_percentage.style.format("{:.1f}%")
        st.table(formatted_pivot_table_percentage)

    else:
        st.write("Please select a date range and other filters to view analytics.")
        
    if not df_filtered.empty:
        st.markdown("## Day Wise Data")
        st.dataframe(df_filtered, height=300)
    else:
        st.write("df_filtered is empty")
        
    if not df_filtered_tel.empty:
        st.markdown("## SOC Data")
        st.dataframe(df_filtered_tel, height=300)
    else:
        st.write("df_filtered_tel is empty")

    invalid_reg_no_values = [None, np.nan, "NA", "0", "FALSE", "NULL"]
    invalid_telematics_values = [None, np.nan, "111111111111111", "FALSE", "11111111111111", "A"]

    df_filtered_mapping['chassis_number'] = replace_invalid_values(df_filtered_mapping['chassis_number'], 'Unknown Chassis', [None, np.nan, False, 0, '0'])
    df_filtered_mapping['reg_no'] = replace_invalid_values(df_filtered_mapping['reg_no'], 'Unknown Reg', invalid_reg_no_values)
    df_filtered_mapping['telematics_number'] = replace_invalid_values(df_filtered_mapping['telematics_number'], 'Unknown Telematics', invalid_telematics_values)

    df_filtered_mapping.reset_index(drop=True, inplace=True)
    df_filtered_mapping['Row Number'] = df_filtered_mapping.index + 1

    if not df_filtered_mapping.empty:
        st.markdown("## List of Assets")
        st.dataframe(df_filtered_mapping[['Row Number', 'chassis_number', 'reg_no', 'telematics_number', 'location', 'client_name', 'battery_type']], height=300)
    else:
        st.write("No assets found for the selected filters.")

if __name__ == "__main__":
    main()

