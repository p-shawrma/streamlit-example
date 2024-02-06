import streamlit as st
import psycopg2
import pandas as pd
from streamlit_echarts import st_echarts
import numpy as np
import os
from openai import OpenAI
import time
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydeck as pdk


# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Set page configuration to wide mode and set page title
st.set_page_config(layout="wide", page_title="Vehicle Telematics Dashboard")

# Mapbox Access Token
# Set Mapbox access token
px.set_mapbox_access_token("pk.eyJ1IjoicC1zaGFybWEiLCJhIjoiY2xzNjRzbTY1MXNodjJsbXUwcG0wNG50ciJ9.v32bwq-wi6whz9zkn6ecow")

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


    
def main():
    if 'last_refresh' not in st.session_state:
        st.session_state['last_refresh'] = time.time()
    # Check if the refresh interval has passed
    refresh_interval = 3000  # seconds (e.g., 300 seconds = 5 minutes)
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
    
                # Registration Number filter
                reg_nos = df_filtered['reg_no'].dropna().unique().tolist()
                selected_reg_nos = st.multiselect('Registration Number', reg_nos)
    
                # Filter dataframe by registration number if selected
                if selected_reg_nos:
                    df_filtered = df_filtered[df_filtered['reg_no'].isin(selected_reg_nos)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['reg_no'].isin(selected_reg_nos)]
    
                # City filter
                cities = df_filtered['deployed_city'].dropna().unique().tolist()
                selected_cities = st.multiselect('City', cities)
    
                # Filter dataframe by city if selected
                if selected_cities:
                    df_filtered = df_filtered[df_filtered['deployed_city'].isin(selected_cities)]
                    df_filtered_tel = df_filtered_tel[df_filtered_tel['deployed_city'].isin(selected_cities)]



    
    
    # Layout: 2 columns
    col1, col2, col3 = st.columns(3)

    # Column 1 - Average Distance Travelled
    with col1:
        st.markdown("## Average Distance")
        if 'df_filtered' in locals() and not df_filtered.empty:
            
            avg_dist_all_vehicles_per_day = df_filtered.groupby('date')['total_km_travelled'].mean().round(1).reset_index()
            overall_avg_dist_per_day = avg_dist_all_vehicles_per_day['total_km_travelled'].mean().round(1)
            st.metric(" ", f"{overall_avg_dist_per_day:.2f} km")

            # Plotting the average distance travelled for all vehicles per day
            options = {
                "xAxis": {
                    "type": "category",
                    "data": avg_dist_all_vehicles_per_day['date'].dt.strftime('%d-%m-%Y').tolist(),
                    "axisLabel": {
                        "rotate": 90,  # Rotating x-axis labels by 90 degrees
                        "fontSize": 10  # Reducing font size
                    }
                },
                "yAxis": {
                    "type": "value"
                },
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {
                        "type": "cross"
                    }
                },
                "series": [{
                    "data": avg_dist_all_vehicles_per_day['total_km_travelled'].tolist(),
                    "type": "line"
                }]
            }
            st_echarts(options=options, height="400px")


        else:
            st.write("Please select a date range and other filters to view analytics.")
        
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
        df_range = df_filtered[(df_filtered['total_km_travelled'] > 20) & (df_filtered['total_discharge_soc'] < -10)]

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
        st.markdown("## Vehicle level Total Distance travelled and range")
        # st.markdown("## ")
        # st.markdown("## ")
        # st.markdown("## ")
        # Filter df_filtered for total_km_travelled > 15km
        df_range = df_filtered[(df_filtered['total_km_travelled'] > 20) & (df_filtered['total_discharge_soc'] < -10)]
        
        if not df_range.empty:
            # Group by reg_no and calculate the sum of total_km_travelled and the Range
            df_grouped = df_range.groupby('reg_no').agg(
                total_km_travelled_sum=('total_km_travelled', 'sum'),
                total_discharge_soc_sum=('total_discharge_soc', 'sum')
            ).reset_index()
            df_grouped['Range'] = df_grouped['total_km_travelled_sum'] * (-100) / df_grouped['total_discharge_soc_sum']

            # Format the DataFrame to match the screenshot layout
            df_display = df_grouped[['reg_no', 'total_km_travelled_sum', 'Range']].rename(
                columns={'reg_no': 'Registration Number', 'total_km_travelled_sum': 'Total KM Travelled'}
            )
            

            
            st.dataframe(df_display,height=400)
            
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


        
    # Display the filtered dataframe below the charts
    if not df_filtered.empty:
        st.markdown("## Day Wise Data")
        st.dataframe(df_range, height=300)


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
