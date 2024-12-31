import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import clickhouse_connect
from haversine import haversine, Unit
from geopy.geocoders import Nominatim
import psycopg2
from pygwalker.api.streamlit import StreamlitRenderer

# @st.cache_data(ttl=6000)
# def fetch_mapping_table():
#     user = "postgres.gqmpfexjoachyjgzkhdf"
#     password = "Change@2015Log9"
#     host = "aws-0-ap-south-1.pooler.supabase.com"
#     port = "5432"
#     dbname = "postgres"
    
#     with psycopg2.connect(
#         dbname=dbname,
#         user=user,
#         password=password,
#         host=host,
#         port=port
#     ) as conn:
#         cursor = conn.cursor()
#         query = "SELECT * FROM mapping_table"
#         cursor.execute(query)
#         records = cursor.fetchall()
#         columns = [desc[0] for desc in cursor.description]
#         df_mapping = pd.DataFrame(records, columns=columns)
#         return df_mapping

ch_host = 'a84a1hn9ig.ap-south-1.aws.clickhouse.cloud'
ch_user = 'default'
ch_password = 'dKd.Y9kFMv06x'
ch_database = 'landing_zone_telematics'

def create_client():
    return clickhouse_connect.get_client(
        host=ch_host,
        user=ch_user,
        password=ch_password,
        database=ch_database,
        secure=True
    )
    
@st.cache_data(ttl=6000)
def fetch_model_numbers_and_dates():
    client = create_client()
    query = "SELECT DISTINCT model_number, toStartOfDay(DeviceDate) as DeviceDate FROM consolidated_custom_report"
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df

@st.cache_data(ttl=6000)
def fetch_mapping_table():
    client = create_client()
    query = "SELECT * FROM mapping_table"
    result = client.query(query)
    df_mapping = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df_mapping

@st.cache_data(ttl=6000)
def fetch_data(model_numbers, start_date, end_date):
    if not model_numbers:
        return pd.DataFrame()  # Return empty DataFrame if no model numbers are selected

    client = create_client()
    model_numbers_str = ','.join([f"'{model}'" for model in model_numbers])
    start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    query = f"""
    SELECT *
    FROM consolidated_custom_report
    WHERE model_number IN ({model_numbers_str}) AND DeviceDate BETWEEN '{start_date}' AND '{end_date}'
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df
    
def haversine_distance(lat1, lon1, lat2, lon2):
    loc1 = (lat1, lon1)
    loc2 = (lat2, lon2)
    return haversine(loc1, loc2, unit=Unit.KILOMETERS)

def get_location_description(lat, lon):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse((lat, lon), language='en')
    return location.address if location else "Unknown location"

# def process_data(df):
#     grouped = df.groupby('model_number')
#     processed_dfs = []

#     for name, group in grouped:
#         group['timestamp'] = pd.to_datetime(group['DeviceDate'])
#         group = group.sort_values(by='timestamp')
#         group['time_diff'] = group['timestamp'].diff().dt.total_seconds()

#         group['time_diff'].fillna(method='bfill', inplace=True)

#         # Define outlier thresholds
#         current_outlier_upper = 1000
#         current_outlier_lower = -1000
#         voltage_upper = 100
#         voltage_lower = 0
#         soc_upper = 100
#         soc_lower = 0

#         # Remove outliers
#         group = group[(group['BM_BattCurrrent'] <= current_outlier_upper) & (group['BM_BattCurrrent'] >= current_outlier_lower)]
#         group = group[(group['BM_BattVoltage'] <= voltage_upper) & (group['BM_BattVoltage'] >= voltage_lower)]
#         group = group[(group['BM_SocPercent'] <= soc_upper) & (group['BM_SocPercent'] >= soc_lower)]

#         base_time_diff = 10
#         base_alpha = 0.33

#         group['alpha'] = group['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
#         group['alpha'] = group['alpha'].clip(upper=0.66)

#         ema_current = group['BM_BattCurrrent'].iloc[0]
#         smoothed_currents = [ema_current]

#         for i in range(1, len(group)):
#             alpha = group['alpha'].iloc[i]
#             current = group['BM_BattCurrrent'].iloc[i]
#             ema_current = ema_current * (1 - alpha) + current * alpha
#             smoothed_currents.append(ema_current)

#         group['Fitted_Current(A)'] = smoothed_currents
#         group['Fitted_Voltage(V)'] = group['BM_BattVoltage'].ewm(alpha=base_alpha).mean()

#         group['voltage_increase'] = group['Fitted_Voltage(V)'].diff() >= 0.05
#         group['soc_increase'] = group['BM_SocPercent'].diff() >= 0.05

#         cell_temp_columns = [col for col in group.columns if 'Cell_Temperature' in col]
#         group['Pack_Temperature_(C)'] = group[cell_temp_columns].mean(axis=1)

#         epsilon = 0.5
#         conditions = [
#             (group['Fitted_Current(A)'] > epsilon) | (group['voltage_increase'] | group['soc_increase']),
#             (group['Fitted_Current(A)'] < -epsilon) & ~((group['voltage_increase']) | (group['soc_increase'])),
#             abs(group['Fitted_Current(A)']) <= epsilon
#         ]
#         choices = ['Charging', 'Discharging', 'Idle']
#         group['state'] = np.select(conditions, choices, default='Idle')

#         group['state_change'] = (group['state'] != group['state'].shift(1)).cumsum()
#         grp = group.groupby('state_change')
#         group['state_duration'] = grp['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
#         group['soc_diff'] = grp['BM_SocPercent'].transform(lambda x: x.iloc[-1] - x.iloc[0])

#         # Applying the new condition for filtered state
#         group['filtered_state'] = np.where(
#             ((group['state'] == 'Idle') & (group['state_duration'] > 600)) | 
#             ((group['state'] != 'Idle') & (group['state_duration'] > 30)), 
#             group['state'], 
#             np.nan
#         )
#         group['filtered_state'].fillna(method='ffill', inplace=True)

#         group['final_state'] = np.where(
#             (group['soc_diff'].abs() <= 1) & (group['filtered_state'] == 'Charging'),
#             np.nan,
#             group['filtered_state']
#         )
#         group['final_state'].fillna(method='ffill', inplace=True)

#         state_mapping = {'Charging': 0, 'Discharging': 1, 'Idle': 2}
#         group['step_type'] = group['final_state'].map(state_mapping)

#         processed_dfs.append(group)

#     return pd.concat(processed_dfs)
def process_data(df):
    grouped = df.groupby('model_number')
    processed_dfs = []

    for name, group in grouped:
        group['timestamp'] = pd.to_datetime(group['DeviceDate'])
        group = group.sort_values(by='timestamp')
        group['time_diff'] = group['timestamp'].diff().dt.total_seconds()

        group['time_diff'].fillna(method='bfill', inplace=True)

        # Define outlier thresholds
        current_outlier_upper = 1000
        current_outlier_lower = -1000
        voltage_upper = 100
        voltage_lower = 0
        soc_upper = 100
        soc_lower = 0

        # Remove outliers
        group = group[(group['BM_BattCurrrent'] <= current_outlier_upper) & (group['BM_BattCurrrent'] >= current_outlier_lower)]
        group = group[(group['BM_BattVoltage'] <= voltage_upper) & (group['BM_BattVoltage'] >= voltage_lower)]
        group = group[(group['BM_SocPercent'] <= soc_upper) & (group['BM_SocPercent'] >= soc_lower)]

        # Remove rows where the change in BM_SocPercent between two rows is more than 3% or BM_BattVoltage is more than 1.5
        group['soc_diff'] = group['BM_SocPercent'].diff().abs()
        group['voltage_diff'] = group['BM_BattVoltage'].diff().abs()
        group = group[(group['soc_diff'] <= 3) & (group['voltage_diff'] <= 1.5)]

        base_time_diff = 10
        base_alpha = 0.33

        group['alpha'] = group['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
        group['alpha'] = group['alpha'].clip(upper=0.66)

        ema_current = group['BM_BattCurrrent'].iloc[0]
        smoothed_currents = [ema_current]

        for i in range(1, len(group)):
            alpha = group['alpha'].iloc[i]
            current = group['BM_BattCurrrent'].iloc[i]
            ema_current = ema_current * (1 - alpha) + current * alpha
            smoothed_currents.append(ema_current)

        group['Fitted_Current(A)'] = smoothed_currents
        group['Fitted_Voltage(V)'] = group['BM_BattVoltage'].ewm(alpha=base_alpha).mean()

        group['voltage_increase'] = group['Fitted_Voltage(V)'].diff() >= 0.05
        group['soc_increase'] = group['BM_SocPercent'].diff() >= 0.05

        cell_temp_columns = [col for col in group.columns if 'Cell_Temperature' in col]
        group['Pack_Temperature_(C)'] = group[cell_temp_columns].mean(axis=1)

        epsilon = 0.5
        conditions = [
            (group['Fitted_Current(A)'] > epsilon) | (group['voltage_increase'] | group['soc_increase']),
            (group['Fitted_Current(A)'] < -epsilon) & ~((group['voltage_increase']) | (group['soc_increase'])),
            abs(group['Fitted_Current(A)']) <= epsilon
        ]
        choices = ['Charging', 'Discharging', 'Idle']
        group['state'] = np.select(conditions, choices, default='Idle')

        group['state_change'] = (group['state'] != group['state'].shift(1)).cumsum()
        grp = group.groupby('state_change')
        group['state_duration'] = grp['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
        group['soc_diff'] = grp['BM_SocPercent'].transform(lambda x: x.iloc[-1] - x.iloc[0])

        # Applying the new condition for filtered state
        group['filtered_state'] = np.where(
            ((group['state'] == 'Idle') & (group['state_duration'] > 600)) | 
            ((group['state'] != 'Idle') & (group['state_duration'] > 30)), 
            group['state'], 
            np.nan
        )
        group['filtered_state'].fillna(method='ffill', inplace=True)

        group['final_state'] = np.where(
            (group['soc_diff'].abs() <= 1) & (group['filtered_state'] == 'Charging'),
            np.nan,
            group['filtered_state']
        )
        group['final_state'].fillna(method='ffill', inplace=True)
        group['final_state'].fillna('Idle', inplace=True)  # Set remaining nulls to 'Idle'

        state_mapping = {'Charging': 0, 'Discharging': 1, 'Idle': 2}
        group['step_type'] = group['final_state'].map(state_mapping)

        # Calculate distance between rows with final_state as 'Discharging'
        if 'Latitude' in group.columns and 'Longitude' in group.columns:
            group['distance'] = 0  # Initialize distance column
            discharging_rows = group[group['final_state'] == 'Discharging']
            discharging_rows['prev_Latitude'] = discharging_rows['Latitude'].shift()
            discharging_rows['prev_Longitude'] = discharging_rows['Longitude'].shift()
            discharging_rows['distance'] = discharging_rows.apply(
                lambda row: haversine((row['Latitude'], row['Longitude']),
                                      (row['prev_Latitude'], row['prev_Longitude'])) if pd.notnull(row['prev_Latitude']) and pd.notnull(row['prev_Longitude']) and row['Latitude'] != 0 and row['Longitude'] != 0 and row['prev_Latitude'] != 0 and row['prev_Longitude'] != 0 else 0,
                axis=1
            )
            # Set distance to 0 if greater than 1.5 km
            discharging_rows['distance'] = discharging_rows['distance'].apply(lambda x: x if x <= 0.25 else 0)

            # Ensure distance for the first row of each new Discharging group is set to 0
            first_discharge_rows = discharging_rows.index[discharging_rows['state_change'] != discharging_rows['state_change'].shift()]
            group.loc[first_discharge_rows, 'distance'] = 0

            group.loc[discharging_rows.index, 'distance'] = discharging_rows['distance']

        processed_dfs.append(group)

    return pd.concat(processed_dfs)

    return pd.concat(processed_dfs)

    
def calculate_percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def process_grouped_data(df):
    # Group by model_number and the condition change
    grouped = df.groupby(['model_number', (df['final_state'] != df['final_state'].shift()).cumsum()])
    
    # Perform aggregation
    result = grouped.agg(
        model_number=('model_number', 'first'),  # Ensure model_number is retained
        start_timestamp=('timestamp', 'min'),
        end_timestamp=('timestamp', 'max'),
        step_type=('final_state', 'first'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        soc_start=('BM_SocPercent', 'first'),
        soc_end=('BM_SocPercent', 'last'),
        voltage_start=('BM_BattVoltage', 'first'),
        voltage_end=('BM_BattVoltage', 'last'),
        average_current=('BM_BattCurrrent', 'mean'),
        median_current=('BM_BattCurrrent', 'median'),
        min_current=('BM_BattCurrrent', calculate_percentile(5)),
        max_current=('BM_BattCurrrent', calculate_percentile(95)),
        current_25th=('BM_BattCurrrent', calculate_percentile(25)),
        current_75th=('BM_BattCurrrent', calculate_percentile(75)),
        median_max_cell_temperature=('Max_monomer_temperature', 'median'),
        median_min_cell_temperature=('Min_monomer_temperature', 'median'),
        median_pack_temperature=('Pack_Temperature_(C)', 'median')
    )

    result['date'] = result['start_timestamp'].dt.date
    result['change_in_soc'] = result['soc_end'] - result['soc_start']

    columns_ordered = ['model_number', 'date', 'start_timestamp', 'end_timestamp', 'step_type', 'duration_minutes',
                       'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
                       'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
                       'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 'median_pack_temperature']
    result = result.reset_index(drop=True)[columns_ordered]

    # Rename model_number to vehicle_number
    result.rename(columns={
        'model_number': 'vehicle_number',
        'start_timestamp': 'start_time',
        'end_timestamp': 'end_time',
    }, inplace=True)

    return result

# def generate_soc_report(processed_df):
#     # Group by model_number and the condition change
#     grouped = processed_df.groupby(['model_number', (processed_df['final_state'] != processed_df['final_state'].shift()).cumsum()])
    
#     # Perform aggregation
#     result = grouped.agg(
#         model_number=('model_number', 'first'),
#         start_timestamp=('timestamp', 'min'),
#         end_timestamp=('timestamp', 'max'),
#         soc_type=('final_state', 'first'),
#         duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
#         soc_start=('BM_SocPercent', 'first'),
#         soc_end=('BM_SocPercent', 'last'),
#         voltage_start=('BM_BattVoltage', 'first'),
#         voltage_end=('BM_BattVoltage', 'last'),
#         average_current=('BM_BattCurrrent', 'mean'),
#         median_current=('BM_BattCurrrent', 'median'),
#         min_current=('BM_BattCurrrent', lambda x: np.percentile(x, 5)),
#         max_current=('BM_BattCurrrent', lambda x: np.percentile(x, 95)),
#         current_25th=('BM_BattCurrrent', lambda x: np.percentile(x, 25)),
#         current_75th=('BM_BattCurrrent', lambda x: np.percentile(x, 75)),
#         median_max_cell_temperature=('Max_monomer_temperature', 'median'),
#         median_min_cell_temperature=('Min_monomer_temperature', 'median'),
#         median_pack_temperature=('Pack_Temperature_(C)', 'median')
#     )

#     result['start_date'] = result['start_timestamp'].dt.date
#     result['change_in_soc'] = result['soc_end'] - result['soc_start']
#     result['end_date'] = result['end_timestamp'].dt.date

#     result.rename(columns={
#         'model_number': 'vehicle_number',
#         'start_timestamp': 'start_time',
#         'end_timestamp': 'end_time'
#     }, inplace=True)

#     result['primary_id'] = result.apply(lambda row: f"{row['vehicle_number']}-{row['start_time']}", axis=1)
#     result['soc_range'] = result.apply(lambda row: f"{row['soc_start']}% - {row['soc_end']}%", axis=1)

#     # Initialize new columns
#     result['total_distance_km'] = None
#     result['total_running_time_seconds'] = None
#     result['energy_consumption'] = None
#     result['total_halt_time_seconds'] = None
#     result['charging_location'] = None
#     result['charging_location_coordinates'] = None
#     result['charging_type'] = None  # Initialize the new column

#     for idx, row in result.iterrows():
#         relevant_rows = processed_df[(processed_df['model_number'] == row['vehicle_number']) &
#                                      (processed_df['timestamp'] >= row['start_time']) &
#                                      (processed_df['timestamp'] <= row['end_time'])]

#         if row['soc_type'] == 'Discharging':
#             total_distance = 0
#             for i in range(1, len(relevant_rows)):
#                 coord1 = (relevant_rows.iloc[i-1]['Latitude'], relevant_rows.iloc[i-1]['Longitude'])
#                 coord2 = (relevant_rows.iloc[i]['Latitude'], relevant_rows.iloc[i]['Longitude'])
#                 total_distance += haversine(coord1, coord2)
#             result.at[idx, 'total_distance_km'] = total_distance
#             total_running_time = (relevant_rows['timestamp'].max() - relevant_rows['timestamp'].min()).total_seconds()
#             result.at[idx, 'total_running_time_seconds'] = total_running_time
#         elif row['soc_type'] == 'Charging':
#             lat, lon = relevant_rows.iloc[0]['Latitude'], relevant_rows.iloc[0]['Longitude']
#             # result.at[idx, 'charging_location'] = get_location_description(lat, lon)
#             result.at[idx, 'charging_location_coordinates'] = f"{lat}, {lon}"
#             current_75th = row['current_75th']
#             if current_75th < 100:
#                 result.at[idx, 'charging_type'] = "SLOW CHARGING"
#             else:
#                 result.at[idx, 'charging_type'] = "FAST CHARGING"
#         elif row['soc_type'] == 'Idle':
#             total_halt_time = (relevant_rows['timestamp'].max() - relevant_rows['timestamp'].min()).total_seconds()
#             result.at[idx, 'total_halt_time_seconds'] = total_halt_time

#     mapping_df = fetch_mapping_table()
#     mapping_dict = mapping_df.set_index('telematics_number').T.to_dict()

#     def fetch_mapping_info(telematics_number, key):
#         return mapping_dict.get(telematics_number, {}).get(key, None)

#     result['telematics_number'] = result['vehicle_number'].apply(lambda x: x.replace("it_", ""))
#     result['partner_id'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'client_name'))
#     result['product'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'battery_type'))
#     result['deployed_city'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'location'))
#     result['reg_no'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'reg_no'))
#     result['chassis_number'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'chassis_number'))
    
#     columns_ordered = [
#         'vehicle_number', 'start_time', 'end_time', 'soc_type', 'duration_minutes', 
#         'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
#         'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
#         'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 
#         'median_pack_temperature', 'start_date', 'end_date', 'primary_id', 'soc_range', 
#         'total_distance_km', 'total_running_time_seconds', 'total_halt_time_seconds',
#         'charging_location', 'charging_location_coordinates', 'charging_type',  # Include the new column
#         'telematics_number', 'partner_id', 'product', 'deployed_city', 'reg_no', 
#         'chassis_number', 'energy_consumption'
#     ]

#     result = result.reset_index(drop=True)[columns_ordered]
    
#     return result

def generate_soc_report(processed_df):
    # Group by model_number and the condition change
    grouped = processed_df.groupby(['model_number', (processed_df['final_state'] != processed_df['final_state'].shift()).cumsum()])
    
    # Perform aggregation
    result = grouped.agg(
        model_number=('model_number', 'first'),
        start_timestamp=('timestamp', 'min'),
        end_timestamp=('timestamp', 'max'),
        soc_type=('final_state', 'first'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        soc_start=('BM_SocPercent', 'first'),
        soc_end=('BM_SocPercent', 'last'),
        voltage_start=('BM_BattVoltage', 'first'),
        voltage_end=('BM_BattVoltage', 'last'),
        average_current=('BM_BattCurrrent', 'mean'),
        median_current=('BM_BattCurrrent', 'median'),
        min_current=('BM_BattCurrrent', lambda x: np.percentile(x, 5)),
        max_current=('BM_BattCurrrent', lambda x: np.percentile(x, 95)),
        current_25th=('BM_BattCurrrent', lambda x: np.percentile(x, 25)),
        current_75th=('BM_BattCurrrent', lambda x: np.percentile(x, 75)),
        median_max_cell_temperature=('Max_monomer_temperature', 'median'),
        median_min_cell_temperature=('Min_monomer_temperature', 'median'),
        median_pack_temperature=('Pack_Temperature_(C)', 'median')
    )

    result['start_date'] = result['start_timestamp'].dt.date
    result['change_in_soc'] = result['soc_end'] - result['soc_start']
    result['end_date'] = result['end_timestamp'].dt.date

    result.rename(columns={
        'model_number': 'vehicle_number',
        'start_timestamp': 'start_time',
        'end_timestamp': 'end_time'
    }, inplace=True)

    result['primary_id'] = result.apply(lambda row: f"{row['vehicle_number']}-{row['start_time']}", axis=1)
    result['soc_range'] = result.apply(lambda row: f"{row['soc_start']}% - {row['soc_end']}%", axis=1)

    # Initialize new columns
    result['total_distance_km'] = None
    result['total_running_time_seconds'] = None
    result['energy_consumption'] = None
    result['total_halt_time_seconds'] = None
    result['charging_location'] = None
    result['charging_location_coordinates'] = None
    result['charging_type'] = None  # Initialize the new column

    for idx, row in result.iterrows():
        relevant_rows = processed_df[(processed_df['model_number'] == row['vehicle_number']) &
                                     (processed_df['timestamp'] >= row['start_time']) &
                                     (processed_df['timestamp'] <= row['end_time'])]

        if row['soc_type'] == 'Discharging':
            total_distance = 0
            for i in range(1, len(relevant_rows)):
                coord1 = (relevant_rows.iloc[i-1]['Latitude'], relevant_rows.iloc[i-1]['Longitude'])
                coord2 = (relevant_rows.iloc[i]['Latitude'], relevant_rows.iloc[i]['Longitude'])
                if (coord1[0] != 0 and coord1[1] != 0 and coord2[0] != 0 and coord2[1] != 0):
                    distance = haversine(coord1, coord2)
                    if distance <= 0.25:
                        total_distance += distance
                    else:
                        total_distance += 0
                else:
                    total_distance += 0
            result.at[idx, 'total_distance_km'] = total_distance
            total_running_time = (relevant_rows['timestamp'].max() - relevant_rows['timestamp'].min()).total_seconds()
            result.at[idx, 'total_running_time_seconds'] = total_running_time

            # Ensure distance for the first row of each new Discharging group is set to 0
            first_discharge_rows = relevant_rows.index[relevant_rows['final_state'] != relevant_rows['final_state'].shift()]
            processed_df.loc[first_discharge_rows, 'distance'] = 0

        elif row['soc_type'] == 'Charging':
            lat, lon = relevant_rows.iloc[0]['Latitude'], relevant_rows.iloc[0]['Longitude']
            result.at[idx, 'charging_location_coordinates'] = f"{lat}, {lon}"
            current_75th = row['current_75th']
            if current_75th < 100:
                result.at[idx, 'charging_type'] = "SLOW CHARGING"
            else:
                result.at[idx, 'charging_type'] = "FAST CHARGING"
        elif row['soc_type'] == 'Idle':
            total_halt_time = (relevant_rows['timestamp'].max() - relevant_rows['timestamp'].min()).total_seconds()
            result.at[idx, 'total_halt_time_seconds'] = total_halt_time

    mapping_df = fetch_mapping_table()
    mapping_dict = mapping_df.set_index('telematics_number').T.to_dict()

    def fetch_mapping_info(telematics_number, key):
        return mapping_dict.get(telematics_number, {}).get(key, None)

    result['telematics_number'] = result['vehicle_number'].apply(lambda x: x.replace("it_", ""))
    result['partner_id'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'client_name'))
    result['product'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'battery_type'))
    result['deployed_city'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'location'))
    result['reg_no'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'reg_no'))
    result['chassis_number'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'chassis_number'))
    
    columns_ordered = [
        'vehicle_number', 'start_time', 'end_time', 'soc_type', 'duration_minutes', 
        'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
        'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
        'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 
        'median_pack_temperature', 'start_date', 'end_date', 'primary_id', 'soc_range', 
        'total_distance_km', 'total_running_time_seconds', 'total_halt_time_seconds',
        'charging_location', 'charging_location_coordinates', 'charging_type',  # Include the new column
        'telematics_number', 'partner_id', 'product', 'deployed_city', 'reg_no', 
        'chassis_number', 'energy_consumption'
    ]

    result = result.reset_index(drop=True)[columns_ordered]
    
    return result
    
def apply_filters(df):
    step_types = df['step_type'].unique().tolist()
    selected_types = st.sidebar.multiselect('Select Step Types', step_types, default=step_types)
    filtered_df = df[df['step_type'].isin(selected_types)]
    return filtered_df
    
def create_day_wise_summary(df):
    Discharging = df[df['step_type'] == 'Discharging']
    Charging = df[df['step_type'] == 'Charging']

    Discharging_summary = Discharging.groupby(['vehicle_number', 'date']).agg({
        'change_in_soc': 'sum',
        'duration_minutes': ['sum', 'min', 'max', 'median', calculate_percentile(25), calculate_percentile(75)]
    })

    Charging_summary = Charging.groupby(['vehicle_number', 'date']).agg({
        'change_in_soc': 'sum'
    })

    Discharging_summary.columns = ['_'.join(col).strip() for col in Discharging_summary.columns.values]
    Charging_summary.columns = ['total_Charging_soc']

    day_wise_summary = pd.merge(Discharging_summary, Charging_summary, on=['vehicle_number', 'date'], how='outer')
    day_wise_summary.rename(columns={
        'change_in_soc_sum': 'total_Discharging_soc',
        'duration_minutes_sum': 'total_Discharging_time',
        'duration_minutes_min': 'Discharging_time_min',
        'duration_minutes_max': 'Discharging_time_max',
        'duration_minutes_median': 'Discharging_time_median',
        'duration_minutes_percentile_25': 'Discharging_time_25th',
        'duration_minutes_percentile_75': 'Discharging_time_75th'
    }, inplace=True)

    return day_wise_summary
    
def main():
    st.set_page_config(layout="wide", page_title="Battery Discharging Analysis")

    with st.sidebar:
        st.title("Filter Settings")
        # model_numbers_and_dates = fetch_model_numbers_and_dates()
        # model_numbers = model_numbers_and_dates['model_number'].unique().tolist()
        # selected_model_numbers = st.multiselect('Select Model Numbers', model_numbers)
    
        # date_range = model_numbers_and_dates['DeviceDate'].unique()
        # start_date = st.date_input("Start Date", min(date_range), min_value=min(date_range), max_value=max(date_range))
        # end_date = st.date_input("End Date", max(date_range), min_value=min(date_range), max_value=max(date_range))
    
        # # Convert start and end dates to datetime with proper times
        # start_date = datetime.combine(start_date, datetime.min.time())
        # end_date = datetime.combine(end_date, datetime.max.time())

        # fetch_button = st.button("Fetch Data")

        model_numbers_and_dates = fetch_model_numbers_and_dates()
        model_numbers = model_numbers_and_dates['model_number'].unique().tolist()
        selected_model_numbers = st.multiselect('Select Model Numbers', model_numbers)
    
        date_range = model_numbers_and_dates['DeviceDate'].unique()
        start_date_default = max(date_range) - timedelta(days=10)  # Default to last 10 days from End Date
        
        start_date = st.date_input("Start Date", start_date_default, min_value=min(date_range), max_value=max(date_range))
        end_date = st.date_input("End Date", max(date_range), min_value=min(date_range), max_value=max(date_range))
    
        # Convert start and end dates to datetime with proper times
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())

        fetch_button = st.button("Fetch Data")
        
    if fetch_button or 'data_loaded' not in st.session_state:
        if start_date > end_date:
            st.error("End date must be after start date.")
            return

        df = fetch_data(selected_model_numbers, start_date, end_date)
        if not df.empty:
            processed_df = process_data(df)
            grouped_df = process_grouped_data(processed_df)
            soc_report = generate_soc_report(processed_df)  # Generate the SOC report
            st.session_state['processed_df'] = processed_df
            st.session_state['grouped_df'] = grouped_df
            st.session_state['soc_report'] = soc_report  # Save the SOC report
            st.session_state['data_loaded'] = True
        else:
            st.write("No data found for the selected date range.")
            st.session_state['data_loaded'] = False

    if st.session_state.get('data_loaded', False):
        all_step_types = st.session_state['grouped_df']['step_type'].unique().tolist()
        selected_step_types = st.multiselect('Select Step Type', all_step_types, default=all_step_types)

        filtered_df = st.session_state['grouped_df'][st.session_state['grouped_df']['step_type'].isin(selected_step_types)]

        display_data_and_plots(filtered_df, st.session_state['processed_df'], st.session_state['soc_report'])  # Pass the SOC report


def display_data_and_plots(filtered_df, processed_df,soc_report):
    st.write("Data Overview:")
    st.dataframe(processed_df)
    # Add vis spec here
    vis_spec = r"""{"config":[{"config":{"defaultAggregated":false,"geoms":["circle"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0},"encodings":{"dimensions":[{"fid":"model_number","name":"model_number","basename":"model_number","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"DeviceDate","name":"DeviceDate","basename":"DeviceDate","semanticType":"temporal","analyticType":"dimension","offset":0},{"fid":"Latitude","name":"Latitude","basename":"Latitude","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"Longitude","name":"Longitude","basename":"Longitude","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"BM_MaxCellID","name":"BM_MaxCellID","basename":"BM_MaxCellID","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"BM_MinCellID","name":"BM_MinCellID","basename":"BM_MinCellID","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"Charging_MOS_tube_status","name":"Charging_MOS_tube_status","basename":"Charging_MOS_tube_status","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Discharge_MOS_tube_status","name":"Discharge_MOS_tube_status","basename":"Discharge_MOS_tube_status","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Charge_Discharge_cycles","name":"Charge_Discharge_cycles","basename":"Charge_Discharge_cycles","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"timestamp","name":"timestamp","basename":"timestamp","semanticType":"temporal","analyticType":"dimension","offset":0},{"fid":"voltage_increase","name":"voltage_increase","basename":"voltage_increase","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"soc_increase","name":"soc_increase","basename":"soc_increase","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"state","name":"state","basename":"state","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"filtered_state","name":"filtered_state","basename":"filtered_state","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"final_state","name":"final_state","basename":"final_state","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"step_type","name":"step_type","basename":"step_type","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"fid":"BM_BattVoltage","name":"BM_BattVoltage","basename":"BM_BattVoltage","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"BM_BattCurrrent","name":"BM_BattCurrrent","basename":"BM_BattCurrrent","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"analyticType":"measure","fid":"gw_61IE","name":"Delta_Voltage","semanticType":"quantitative","computed":true,"expression":{"op":"expr","as":"gw_61IE","params":[{"type":"sql","value":"BM_MaxCellVolt - BM_MinCellVolt"}]}},{"fid":"BM_SocPercent","name":"BM_SocPercent","basename":"BM_SocPercent","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"BM_MaxCellVolt","name":"BM_MaxCellVolt","basename":"BM_MaxCellVolt","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"BM_MinCellVolt","name":"BM_MinCellVolt","basename":"BM_MinCellVolt","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"Max_monomer_temperature","name":"Max_monomer_temperature","basename":"Max_monomer_temperature","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"Min_monomer_temperature","name":"Min_monomer_temperature","basename":"Min_monomer_temperature","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"BM_BatteryCapacityAh","name":"BM_BatteryCapacityAh","basename":"BM_BatteryCapacityAh","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"time_diff","name":"time_diff","basename":"time_diff","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"soc_diff","name":"soc_diff","basename":"soc_diff","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"voltage_diff","name":"voltage_diff","basename":"voltage_diff","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"alpha","name":"alpha","basename":"alpha","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"Fitted_Current(A)","name":"Fitted_Current(A)","basename":"Fitted_Current(A)","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"Fitted_Voltage(V)","name":"Fitted_Voltage(V)","basename":"Fitted_Voltage(V)","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"Pack_Temperature_(C)","name":"Pack_Temperature_(C)","basename":"Pack_Temperature_(C)","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"state_change","name":"state_change","basename":"state_change","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"state_duration","name":"state_duration","basename":"state_duration","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"distance","name":"distance","basename":"distance","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"fid":"distance","name":"distance","basename":"distance","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"fid":"BM_SocPercent","name":"BM_SocPercent","basename":"BM_SocPercent","analyticType":"measure","semanticType":"quantitative","aggName":"sum","offset":0},{"analyticType":"measure","fid":"gw_61IE","name":"Delta_Voltage","semanticType":"quantitative","computed":true,"expression":{"op":"expr","as":"gw_61IE","params":[{"type":"sql","value":"BM_MaxCellVolt - BM_MinCellVolt"}]}}],"columns":[{"fid":"DeviceDate","name":"DeviceDate","basename":"DeviceDate","semanticType":"temporal","analyticType":"dimension","offset":0}],"color":[{"fid":"final_state","name":"final_state","basename":"final_state","semanticType":"nominal","analyticType":"dimension","offset":0}],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"fixed","width":1000,"height":608},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_vGdL","name":"Chart 1"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"transform","transform":[{"key":"gw_61IE","expression":{"op":"expr","as":"gw_61IE","params":[{"type":"sql","value":"(\"BM_MaxCellVolt\" - \"BM_MinCellVolt\")"}]}}]},{"type":"view","query":[{"op":"raw","fields":["DeviceDate","final_state","distance","BM_SocPercent","gw_61IE"]}]}]}],"version":"0.4.8.9"}"""


    # PyG Walker for data exploration
    pyg_app = StreamlitRenderer(processed_df,spec = vis_spec)
    pyg_app.explorer()
    
    st.write("Filtered Grouped Data Overview:")
    st.dataframe(filtered_df)
        
    summary_df = create_day_wise_summary(filtered_df)
    st.write("Day-wise Summary:")
    st.dataframe(summary_df)

    st.write("SOC Report:")
    st.dataframe(soc_report)

if __name__ == "__main__":
    main()
