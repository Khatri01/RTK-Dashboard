import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
import random
from scipy.optimize import dual_annealing
import pyswarms as ps
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RTK Optimization Tool",page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("RTK-Optimization")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
# Display a note or instructions
st.write("**Note:** The Excel file should have the following format:")
st.markdown("""
- **Column 1:** Time steps
- **Column 2:** Observed RDII
- **Column 3:** Rainfall
""")

if uploaded_file is not None:
    # Check if the file has already been uploaded and stored
    if "uploaded_file_data" not in st.session_state or st.session_state.filename != uploaded_file.name:
        try:
            # Read the uploaded file
            data = pd.read_excel(uploaded_file, skiprows=0)  # Adjust `skiprows` as needed

            # Extract columns
            rainfall = data.iloc[:, 2].dropna().tolist()  # Assuming rainfall is in the third column
            obs_rdii = data.iloc[:, 1].tolist()  # Assuming obs_rdii is in the second column

            # Store data and filename in session state
            st.session_state.uploaded_file_data = data
            st.session_state.rainfall = rainfall
            st.session_state.obs_rdii = obs_rdii
            st.session_state.filename = uploaded_file.name

            st.success("File uploaded and data stored successfully!")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

# Retrieve data from session state if available
if "uploaded_file_data" in st.session_state:
    data = st.session_state.uploaded_file_data
    rainfall = st.session_state.rainfall
    obs_rdii = st.session_state.obs_rdii
    st.write("Using uploaded file:", st.session_state.filename)
    st.write("First few rows of the dataset:")
    st.dataframe(data.head())
else:
    st.warning("Please upload a file to proceed.")

# Persistent Time Step
if "time_step" not in st.session_state:
    st.session_state.time_step = 60  # Default value in minutes
time_step = st.number_input(
    "Time Step [minutes]:",
    min_value=1,
    value=st.session_state.time_step,
    step=1,
    format="%d"
)
st.session_state.time_step = time_step  # Update session state
delta_t = time_step * 60  # Convert to seconds
st.write(f"Time step in seconds (delta_t): {delta_t} seconds")

# Persistent Sewershed Area
if "area_acres" not in st.session_state:
    st.session_state.area_acres = 1.0  # Default value in acres
area_acres = st.number_input(
    "Sewershed Area [acres]:",
    min_value=0.1,
    value=st.session_state.area_acres,
    step=0.1,
    format="%.3f"
)
st.session_state.area_acres = area_acres  # Update session state
st.write(f"Sewershed Area: {area_acres} acres")

def hydrograph_convolution(unit_hydrograph, excess_rainfall):
    """
    Calculates the Direct Runoff Hydrograph (DRH) using convolution.

    Args:
        unit_hydrograph (list ): Unit Hydrograph Ordinates values.
        excess_rainfall (list ): Excess Rainfall Hyetograph values.

    Returns:
        numpy array: Direct Runoff Hydrograph values.
    """

    direct_runoff = np.convolve(unit_hydrograph, excess_rainfall, mode='full')
    return direct_runoff

def unit_hydrograph_ordinates(R, T, K, delta_t):
    '''
    To calculate ordinates of given traingular RTK hydrograph
     Args:
         R= ratio of rdii volume to rainfall volume
         T= time from onset of rainfall to the peak of unit hydrograph (seconds)
         K= ratio of time to recession of the unit hydrograph
         delta_t = rainfall data time step in seconds
    return:
        list of ordinates value of unit hydrographs
    '''
    # Calculate the number of ordinates
    num_ordinates = int((T + K * T) / delta_t)
    
    # Initialize list to store the ordinates
    ordinates = []

    # Loop through each ordinate
    for j in range(1, num_ordinates + 1):
        # Calculate tau
        tau = (j - 0.5) * delta_t
        
        # Determine the value of f based on the conditions
        if tau <= T:
            f = tau / T
        elif T < tau <= T + K * T:
            f = 1 - (tau - T) / (K * T)
        else:
            f = 0

        # Calculate the unit hydrograph ordinate
        UH = 2 * R * f / (T + K * T)
        ordinates.append(UH)

    return ordinates

def plot_unit_hydrographs_with_sum(params, delta_t, filename=None):
    """
    Plots multiple unit hydrographs based on given R, T, K parameter sets and their summation.
    
    Args:
        params (list of tuples): Each tuple contains (R, T, K) values.
        delta_t (float): Time step in seconds.
        filename (str, optional): If provided, saves the plot to the given filename.
    """
    fig, ax = plt.subplots(figsize=(10*0.8, 6*0.8))  # Create figure for Streamlit compatibility
    
    # Store all ordinates and time values
    all_ordinates = []
    max_length = 0  # To find the maximum length of the ordinates

    # Iterate over each set of R, T, K values
    for i, (R, T, K) in enumerate(params, start=1):
        # Calculate the ordinates for the current R, T, K
        ordinates = unit_hydrograph_ordinates(R, T, K, delta_t)
        
        # Add 0 at the beginning and end of the ordinates list
        ordinates = [0] + ordinates + [0]
        
        # Keep track of the maximum length of ordinates
        max_length = max(max_length, len(ordinates))
        
        # Append ordinates to all_ordinates for summation
        all_ordinates.append(ordinates)
        
        # Generate the time values for the x-axis, starting from 0
        time_values = [j * delta_t / 3600 for j in range(len(ordinates))]
        
        # Plot the unit hydrograph for the current R, T, K
        ax.plot(time_values, ordinates, linestyle='-', label=f'UH{i} (R={R}, T={T}, K={K})')

    # Pad the shorter ordinates with zeros to match the maximum length
    padded_ordinates = []
    for ordinates in all_ordinates:
        padded_ordinates.append(ordinates + [0] * (max_length - len(ordinates)))

    # Calculate the summation of all ordinates element-wise
    summed_ordinates = np.sum(padded_ordinates, axis=0)
    
    # Generate the time values for the x-axis for the summed plot
    time_values = [i * delta_t / 3600 for i in range(max_length)]
    
    # Plot the summed unit hydrograph
    ax.plot(time_values, summed_ordinates, linestyle='--', color='black', label='Sum of All UH Sets')
    
    # Plot settings
    ax.set_xlabel('Time [Hrs]')
    ax.set_ylabel('[1/sec]')
    ax.set_title('Unit Hydrographs')
    ax.legend()
    ax.grid(True)
    
    # Set the x and y axis limits to start at 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
         
    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    # Render the plot in Streamlit
    st.pyplot(fig)


def add_flow(*arrays):
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad each array with zeros to match the length of the longest array
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in arrays]
    
    # Sum the padded arrays element-wise
    result = np.sum(padded_arrays, axis=0)
    
    return result.tolist()

def RDII_calculation_and_plot(params, delta_t, rainfall, Area, filename=None, obs_rdii=None):
    # Unpack the tuple of parameters
    (R1, T1, K1), (R2, T2, K2), (R3, T3, K3) = params
    
    # Calculate unit hydrograph ordinates for each set of parameters
    uh1_ordinates = unit_hydrograph_ordinates(R1, T1, K1, delta_t)
    uh2_ordinates = unit_hydrograph_ordinates(R2, T2, K2, delta_t)
    uh3_ordinates = unit_hydrograph_ordinates(R3, T3, K3, delta_t)
    
    # Perform convolution with rainfall data
    Q1_inch_sec = hydrograph_convolution(uh1_ordinates, rainfall)
    Q2_inch_sec = hydrograph_convolution(uh2_ordinates, rainfall)
    Q3_inch_sec = hydrograph_convolution(uh3_ordinates, rainfall)

    # Convert flow from inch/sec to cubic feet per second (cfs)
    Q1_cfs = Q1_inch_sec * Area * 43560 / 12
    Q2_cfs = Q2_inch_sec * Area * 43560 / 12
    Q3_cfs = Q3_inch_sec * Area * 43560 / 12

    # Total flow by adding the flows from each hydrograph
    total_flow = add_flow(Q1_cfs, Q2_cfs, Q3_cfs)

    # Generate the time axis in hours
    time_values = [i * delta_t / 3600 for i in range(len(total_flow))]

    # Handle the case where obs_rdii is provided
    if obs_rdii is not None:
        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(total_flow), len(obs_rdii))
        padded_sim_rdii = np.pad(total_flow, (0, max_length - len(total_flow)), 'constant')
        padded_obs_rdii = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), 'constant')
        padded_time = [i * delta_t / 3600 for i in range(max_length)]  # Extend time to match max_length
    else:
        padded_sim_rdii = total_flow
        padded_obs_rdii = None
        padded_time = time_values

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10*0.8, 6*0.8))

    # Plot total RDII (simulated) on the first y-axis
    ax1.plot(padded_time, padded_sim_rdii, label="Simulated RDII", color='b', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Time (Hrs)')
    ax1.set_ylabel('RDII (cfs)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot observed RDII if available
    if obs_rdii is not None:
        ax1.plot(padded_time, padded_obs_rdii, label="Observed RDII ", color='r', linestyle='-', linewidth=1.5)
        ax1.legend(loc='upper left')

    # Create a secondary y-axis for rainfall
    ax2 = ax1.twinx()
    rainfall_time_values = [i * delta_t / 3600 for i in range(len(rainfall))]
    ax2.bar(rainfall_time_values, rainfall, width=delta_t / 3600 * 0.8, color='gray', alpha=0.5, label="Rainfall")
    ax2.set_ylabel('Rainfall (in)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')

    # Set the y-axis limits to the same range
    y_min = 0
    y_max = max(
        max(total_flow), 
        max(rainfall), 
        max(obs_rdii) if obs_rdii is not None else 0
    ) * 1.2  # Add 20% padding
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Ensure the x-axis starts at 0
    plt.xlim(left=0)

    #plt.title('Total RDII and Rainfall Over Time')

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    st.pyplot(fig)

    return total_flow

def RDII_calculation(params, delta_t, rainfall, Area, filename = None):
   # Unpack the tuple of parameters
    (R1, T1, K1), (R2, T2, K2), (R3, T3, K3) = params
       #Convert flat array back to parameter tuples
    #R1, T1, K1, R2, T2, K2, R3, T3, K3 = params_flat
    
    # R1, T1_scaled, K1, R2, T2_scaled, K2, R3, T3_scaled, K3 = params_flat
    # T1, T2, T3 = T1_scaled * 10000, T2_scaled * 10000, T3_scaled * 10000
    
    # params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]
    
    # Calculate unit hydrograph ordinates for each set of parameters
    uh1_ordinates = unit_hydrograph_ordinates(R1, T1, K1, delta_t)
    uh2_ordinates = unit_hydrograph_ordinates(R2, T2, K2, delta_t)
    uh3_ordinates = unit_hydrograph_ordinates(R3, T3, K3, delta_t)
    
    # Perform convolution with rainfall data
    Q1_inch_sec = hydrograph_convolution(uh1_ordinates, rainfall)
    Q2_inch_sec = hydrograph_convolution(uh2_ordinates, rainfall)
    Q3_inch_sec = hydrograph_convolution(uh3_ordinates, rainfall)

    # Convert flow from inch/sec to cubic feet per second (cfs)
    Q1_cfs = Q1_inch_sec * Area* 43560 / 12
    Q2_cfs = Q2_inch_sec * Area* 43560 / 12
    Q3_cfs = Q3_inch_sec * Area* 43560 / 12

    # Total flow by adding the flows from each hydrograph
    total_flow = add_flow(Q1_cfs, Q2_cfs, Q3_cfs)
    
    
    return total_flow

def R_calc(rainfall_series, rdii_series, delta_t, Area_acres):
    """
    Calculate R, the fraction of RDII volume to total rainfall volume.

    Args:
        rainfall_series (array-like): Time series of rainfall data(inch).
        rdii_series (array-like): Time series of RDII data (cfs).
        delta_t (float): Time step in seconds.
        Area_acres(float) : Area in acres 

    Returns:
        float: The calculated R value.
    """
    # Convert rainfall depth (inches) to volume in cubic feet
    total_rainfall_depth_in = np.sum(rainfall_series)  # Sum of rainfall in inches
    total_rainfall_volume_cf = total_rainfall_depth_in * Area_acres * 3630
       
    # Calculate total rainfall volume and RDII volume
    total_rdii_volume = np.trapz(rdii_series, dx= delta_t) # in 
   
   
    # Calculate R as the ratio of RDII volume to total rainfall volume
    R = total_rdii_volume / total_rainfall_volume_cf if total_rainfall_volume_cf != 0 else 0

    return R

def fitness(obs_rdii, sim_rdii, delta_t, weight_rmse, weight_r2, weight_pbias, weight_nse):
    """
    Calculate a combined fitness score using normalized RMSE, R², Percent Bias (PBIAS), and NSE.

    Args:
        obs_rdii (array-like): Observed RDII values.
        sim_rdii (array-like): Simulated RDII values.
        delta_t (float): Time interval in seconds.
        weight_rmse (float): Weight for RMSE in the combined score.
        weight_r2 (float): Weight for R² penalty in the combined score.
        weight_pbias (float): Weight for Percent Bias in the combined score.
        weight_nse (float): Weight for NSE penalty in the combined score.

    Returns:
        float: Combined fitness score (lower is better).
    """
    # Convert inputs to numpy arrays
    obs_rdii = np.array(obs_rdii)
    sim_rdii = np.array(sim_rdii)
    
    # Pad the shorter array with zeros to match the length of the longer one
    max_length = max(len(obs_rdii), len(sim_rdii))
    obs_rdii = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
    sim_rdii = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')
    
    # Check for invalid outputs
    if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
        print("Invalid sim_rdii generated, returning large penalty.")
        return float('inf')

    # Replace NaNs or infinite values with large penalties
    obs_rdii = np.nan_to_num(obs_rdii, nan=0.0, posinf=1e6, neginf=-1e6)
    sim_rdii = np.nan_to_num(sim_rdii, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((obs_rdii - sim_rdii) ** 2))
    normalized_rmse = rmse / (rmse + 1)  # Normalize RMSE to range [0, 1]
    
    # Calculate R² (Coefficient of Determination)
    ss_total = np.sum((obs_rdii - np.mean(obs_rdii)) ** 2)
    ss_residual = np.sum((obs_rdii - sim_rdii) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    r2_penalty = abs(1 - r2)  # Minimize deviation from 1

    # Calculate Percent Bias (PBIAS)
    pbias = 100 * np.sum(sim_rdii - obs_rdii) / np.sum(obs_rdii) if np.sum(obs_rdii) != 0 else float('inf')
    normalized_pbias = abs(pbias) / (abs(pbias) + 1)  # Normalize PBIAS to range [0, 1]

    # Calculate Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (ss_residual / ss_total) if ss_total != 0 else -float('inf')
    nse_penalty = abs(1 - max(0, nse))  # NSE below 0 is penalized more

    # Combined fitness score (weighted sum of penalties)
    combined_score = (
        weight_rmse * normalized_rmse +
        weight_r2 * r2_penalty +
        weight_pbias * normalized_pbias +
        weight_nse * nse_penalty
    )

    return combined_score

def calculate_criteria(best_params, obs_rdii, delta_t, rainfall, area_acres):
    """
    Calculate and print RMSE, R2, Percent Bias, NSE, and Percent Volume Difference
    for the given best parameters.

    Args:
        best_params (list of tuples): The best parameters as a list of three (R, T, K) tuples.
        obs_rdii (array-like): Observed RDII values.
        delta_t (float): Time interval in seconds.
        rainfall (array-like): Rainfall values.
        area_acres (float): Area in acres.
    """
    # Simulate RDII using the best parameters
    if num_unit_hydrographs==3:
        sim_rdii = RDII_calculation(best_params, delta_t, rainfall, area_acres)
    elif num_unit_hydrographs==2:
        sim_rdii= RDII_calculation_2uh(best_params, delta_t, rainfall, area_acres)
    elif num_unit_hydrographs==4:
        sim_rdii= RDII_calculation_4uh(best_params, delta_t, rainfall, area_acres)

    sim_rdii = np.array(sim_rdii)

    # Convert inputs to numpy arrays
    obs_rdii = np.array(obs_rdii)
        
    # Pad the shorter array with zeros to match the length of the longer one
    max_length = max(len(obs_rdii), len(sim_rdii))
    obs_rdii = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
    sim_rdii = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

    # # Adjust lengths to match obs_rdii
    # if len(sim_rdii) > len(obs_rdii):  # Truncate if sim_rdii is longer
    #     sim_rdii = sim_rdii[:len(obs_rdii)]
    # elif len(sim_rdii) < len(obs_rdii):  # Pad zeros if sim_rdii is shorter
    #     sim_rdii = np.pad(sim_rdii, (0, len(obs_rdii) - len(sim_rdii)), mode='constant')

    # Replace NaNs or infinite values with large penalties
    obs_rdii = np.nan_to_num(obs_rdii, nan=0.0, posinf=1e6, neginf=-1e6)
    sim_rdii = np.nan_to_num(sim_rdii, nan=0.0, posinf=1e6, neginf=-1e6)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((obs_rdii - sim_rdii) ** 2))

    # Calculate R2
    ss_res = np.sum((obs_rdii - sim_rdii) ** 2)
    ss_tot = np.sum((obs_rdii - np.mean(obs_rdii)) ** 2)
    r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else float('inf'))

    # Calculate Percent Bias (PBIAS)
    pbias = 100 * np.sum(sim_rdii - obs_rdii) / np.sum(obs_rdii) if np.sum(obs_rdii) != 0 else float('inf')

    # Calculate NSE
    nse = 1 - (ss_res / ss_tot if ss_tot != 0 else float('inf'))

    # # Calculate Percent Volume Difference
    # obs_volume = np.trapz(obs_rdii, dx=delta_t)
    # sim_volume = np.trapz(sim_rdii, dx=delta_t)
    # vol_diff = 100 * (sim_volume - obs_volume) / obs_volume if obs_volume != 0 else float('inf')

    # Print the criteria
    print(f"Criteria values for best parameters:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Percent Bias (PBIAS): {pbias:.2f}%")
    print(f"  Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
   # print(f"  Percent Volume Difference: {vol_diff:.2f}%")

    return {
        "RMSE": rmse,
        "R2": r2,
        "PBIAS": pbias,
        "NSE": nse
        #"Vol_Diff": vol_diff
    }

def plot_rdii(obs_rdii, sim_rdii, delta_t, filename = None):
    """
    Plot observed and simulated RDII on the same plot.

    Args:
        obs_rdii (array-like): Observed RDII values.
        sim_rdii (array-like): Simulated RDII values.
        delta_t (float): Time interval in seconds.
    """
    # Calculate the time values in minutes
    time_values = np.arange(len(obs_rdii)) * delta_t / 3600  # Convert to minutes

    # Plot observed and simulated RDII
    plt.figure(figsize=(12, 6))
    plt.plot(time_values, obs_rdii, label="Observed RDII", color='blue')
    plt.plot(time_values, sim_rdii, label="Simulated RDII", color='orange')
    
    # Plot settings
    plt.xlabel('Time [Hrs]')
    plt.ylabel('RDII [cfs]')
    plt.title('Observed vs. Simulated RDII')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

        # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

        
    # Show plot
    plt.show()

 # Input for lower and upper limits of T1, T2, T3

def map_index_to_value(index, allowed_values):
    """Map a continuous index to the closest value in the allowed set."""
    index = int(round(index))  # Ensure it's an integer
    index = max(0, min(index, len(allowed_values) - 1))  # Clip to valid range
    return allowed_values[index]

def convert_seconds_to_hm(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours} hrs {minutes} mins"

def objective_function(params_flat):
        # Convert flat array back to parameter tuples
        R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index = params_flat

        # Map indices to actual T and K values
        T1 = allowed_T1_values[int(T1_index)]
        T2 = allowed_T2_values[int(T2_index)]
        T3 = allowed_T3_values[int(T3_index)]

        K1 = allowed_K1_values[int(K1_index)]
        K2 = allowed_K2_values[int(K2_index)]
        K3 = allowed_K3_values[int(K3_index)]

        # Package parameters
        param_tuples = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]

        # Calculate simulated RDII
        sim_rdii = RDII_calculation(param_tuples, delta_t, rainfall, area_acres)

        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # # Calculate fitness (RMSE)
        # rmse = fitness(obs_rdii, sim_rdii, delta_t)
    
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )


        # Calculate penalties
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t,area_acres)

        # R constraints
        if R1 + R2 + R3 > Ro:
            penalty += 1000 * (R1 + R2 + R3 - Ro)

        #    T and K constraints
        if not (T1 < T2 < T3):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3):
            penalty += 1000

        return (fitness_value + penalty,)  # Ensure the result is a tuple for GA algorithms

def map_index_to_value(index, allowed_values):
    index = int(round(index))  # Ensure the index is an integer
    index = max(0, min(index, len(allowed_values) - 1))  # Clip to valid range
    return allowed_values[index]


def RDII_calculation_and_plot_2uh(params, delta_t, rainfall, Area, filename=None, obs_rdii=None):
    # Unpack the tuple of parameters
    (R1, T1, K1), (R2, T2, K2) = params
    
    # Calculate unit hydrograph ordinates for each set of parameters
    uh1_ordinates = unit_hydrograph_ordinates(R1, T1, K1, delta_t)
    uh2_ordinates = unit_hydrograph_ordinates(R2, T2, K2, delta_t)
      
    # Perform convolution with rainfall data
    Q1_inch_sec = hydrograph_convolution(uh1_ordinates, rainfall)
    Q2_inch_sec = hydrograph_convolution(uh2_ordinates, rainfall)
   

    # Convert flow from inch/sec to cubic feet per second (cfs)
    Q1_cfs = Q1_inch_sec * Area * 43560 / 12
    Q2_cfs = Q2_inch_sec * Area * 43560 / 12
    
    # Total flow by adding the flows from each hydrograph
    total_flow = add_flow(Q1_cfs, Q2_cfs)

    # Generate the time axis in hours
    time_values = [i * delta_t / 3600 for i in range(len(total_flow))]

    # Handle the case where obs_rdii is provided
    if obs_rdii is not None:
        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(total_flow), len(obs_rdii))
        padded_sim_rdii = np.pad(total_flow, (0, max_length - len(total_flow)), 'constant')
        padded_obs_rdii = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), 'constant')
        padded_time = [i * delta_t / 3600 for i in range(max_length)]  # Extend time to match max_length
    else:
        padded_sim_rdii = total_flow
        padded_obs_rdii = None
        padded_time = time_values

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot total RDII (simulated) on the first y-axis
    ax1.plot(padded_time, padded_sim_rdii, label="Simulated RDII", color='b', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Time (Hrs)')
    ax1.set_ylabel('RDII (cfs)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot observed RDII if available
    if obs_rdii is not None:
        ax1.plot(padded_time, padded_obs_rdii, label="Observed RDII ", color='r', linestyle='-', linewidth=1.5)
        ax1.legend(loc='upper left')

    # Create a secondary y-axis for rainfall
    ax2 = ax1.twinx()
    rainfall_time_values = [i * delta_t / 3600 for i in range(len(rainfall))]
    ax2.bar(rainfall_time_values, rainfall, width=delta_t / 3600 * 0.8, color='gray', alpha=0.5, label="Rainfall")
    ax2.set_ylabel('Rainfall (in)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')

    # Set the y-axis limits to the same range
    y_min = 0
    y_max = max(
        max(total_flow), 
        max(rainfall), 
        max(obs_rdii) if obs_rdii is not None else 0
    ) * 1.2  # Add 20% padding
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Ensure the x-axis starts at 0
    plt.xlim(left=0)

    plt.title('Total RDII and Rainfall Over Time')

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    st.pyplot(fig)

    return total_flow

def RDII_calculation_2uh(params, delta_t, rainfall, Area, filename = None):
   # Unpack the tuple of parameters
    (R1, T1, K1), (R2, T2, K2)  = params
       
    # Calculate unit hydrograph ordinates for each set of parameters
    uh1_ordinates = unit_hydrograph_ordinates(R1, T1, K1, delta_t)
    uh2_ordinates = unit_hydrograph_ordinates(R2, T2, K2, delta_t)
       
    # Perform convolution with rainfall data
    Q1_inch_sec = hydrograph_convolution(uh1_ordinates, rainfall)
    Q2_inch_sec = hydrograph_convolution(uh2_ordinates, rainfall)
   
    # Convert flow from inch/sec to cubic feet per second (cfs)
    Q1_cfs = Q1_inch_sec * Area* 43560 / 12
    Q2_cfs = Q2_inch_sec * Area* 43560 / 12
    
    # Total flow by adding the flows from each hydrograph
    total_flow = add_flow(Q1_cfs, Q2_cfs)
        
    return total_flow

def RDII_calculation_and_plot_4uh(params, delta_t, rainfall, Area, filename=None, obs_rdii=None):
    # Unpack the tuple of parameters
    (R1, T1, K1), (R2, T2, K2), (R3, T3, K3) , (R4, T4, K4)= params
    
    # Calculate unit hydrograph ordinates for each set of parameters
    uh1_ordinates = unit_hydrograph_ordinates(R1, T1, K1, delta_t)
    uh2_ordinates = unit_hydrograph_ordinates(R2, T2, K2, delta_t)
    uh3_ordinates = unit_hydrograph_ordinates(R3, T3, K3, delta_t)
    uh4_ordinates = unit_hydrograph_ordinates(R4, T4, K4, delta_t)
    
    # Perform convolution with rainfall data
    Q1_inch_sec = hydrograph_convolution(uh1_ordinates, rainfall)
    Q2_inch_sec = hydrograph_convolution(uh2_ordinates, rainfall)
    Q3_inch_sec = hydrograph_convolution(uh3_ordinates, rainfall)
    Q4_inch_sec = hydrograph_convolution(uh4_ordinates, rainfall)

    # Convert flow from inch/sec to cubic feet per second (cfs)
    Q1_cfs = Q1_inch_sec * Area * 43560 / 12
    Q2_cfs = Q2_inch_sec * Area * 43560 / 12
    Q3_cfs = Q3_inch_sec * Area * 43560 / 12
    Q4_cfs = Q4_inch_sec * Area * 43560 / 12

    # Total flow by adding the flows from each hydrograph
    total_flow = add_flow(Q1_cfs, Q2_cfs, Q3_cfs, Q4_cfs)

    # Generate the time axis in hours
    time_values = [i * delta_t / 3600 for i in range(len(total_flow))]

    # Handle the case where obs_rdii is provided
    if obs_rdii is not None:
        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(total_flow), len(obs_rdii))
        padded_sim_rdii = np.pad(total_flow, (0, max_length - len(total_flow)), 'constant')
        padded_obs_rdii = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), 'constant')
        padded_time = [i * delta_t / 3600 for i in range(max_length)]  # Extend time to match max_length
    else:
        padded_sim_rdii = total_flow
        padded_obs_rdii = None
        padded_time = time_values

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot total RDII (simulated) on the first y-axis
    ax1.plot(padded_time, padded_sim_rdii, label="Simulated RDII", color='b', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Time (Hrs)')
    ax1.set_ylabel('RDII (cfs)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot observed RDII if available
    if obs_rdii is not None:
        ax1.plot(padded_time, padded_obs_rdii, label="Observed RDII ", color='r', linestyle='-', linewidth=1.5)
        ax1.legend(loc='upper left')

    # Create a secondary y-axis for rainfall
    ax2 = ax1.twinx()
    rainfall_time_values = [i * delta_t / 3600 for i in range(len(rainfall))]
    ax2.bar(rainfall_time_values, rainfall, width=delta_t / 3600 * 0.8, color='gray', alpha=0.5, label="Rainfall")
    ax2.set_ylabel('Rainfall (in)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')

    # Set the y-axis limits to the same range
    y_min = 0
    y_max = max(
        max(total_flow), 
        max(rainfall), 
        max(obs_rdii) if obs_rdii is not None else 0
    ) * 1.2  # Add 20% padding
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Ensure the x-axis starts at 0
    plt.xlim(left=0)

    plt.title('Total RDII and Rainfall Over Time')

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    st.pyplot(fig)

    return total_flow

def RDII_calculation_4uh(params, delta_t, rainfall, Area, filename = None):
   # Unpack the tuple of parameters
    (R1, T1, K1), (R2, T2, K2), (R3, T3, K3), (R4,T4,K4) = params
     
    # Calculate unit hydrograph ordinates for each set of parameters
    uh1_ordinates = unit_hydrograph_ordinates(R1, T1, K1, delta_t)
    uh2_ordinates = unit_hydrograph_ordinates(R2, T2, K2, delta_t)
    uh3_ordinates = unit_hydrograph_ordinates(R3, T3, K3, delta_t)
    uh4_ordinates = unit_hydrograph_ordinates(R4, T4, K4, delta_t)
    
    # Perform convolution with rainfall data
    Q1_inch_sec = hydrograph_convolution(uh1_ordinates, rainfall)
    Q2_inch_sec = hydrograph_convolution(uh2_ordinates, rainfall)
    Q3_inch_sec = hydrograph_convolution(uh3_ordinates, rainfall)
    Q4_inch_sec = hydrograph_convolution(uh4_ordinates, rainfall)

    # Convert flow from inch/sec to cubic feet per second (cfs)
    Q1_cfs = Q1_inch_sec * Area* 43560 / 12
    Q2_cfs = Q2_inch_sec * Area* 43560 / 12
    Q3_cfs = Q3_inch_sec * Area* 43560 / 12
    Q4_cfs = Q4_inch_sec * Area* 43560 / 12

    # Total flow by adding the flows from each hydrograph
    total_flow = add_flow(Q1_cfs, Q2_cfs, Q3_cfs, Q4_cfs)
    
    
    return total_flow

# Dropdown code
st.subheader("Select Number of Unit Hydrographs")
num_unit_hydrographs = st.selectbox("Number of Unit Hydrographs:", options=[2, 3, 4], index=1)
st.write(f"You selected {num_unit_hydrographs} Unit Hydrographs")

# Conditionally execute blocks for each case
if num_unit_hydrographs == 2:
    st.subheader("Input Time Range for T1 and T2")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### T1")
        st.write("Lower Limit:")
        T1_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T1_lower_hour_2")
        T1_lower_minute = st.number_input("Minutes", min_value=0, value=10, step=1, key="T1_lower_minute_2")
        st.write("Upper Limit:")
        T1_upper_hour = st.number_input("Hours", min_value=0, value=2, step=1, key="T1_upper_hour_2")
        T1_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T1_upper_minute_2")

    with col2:
        st.markdown("### T2")
        st.write("Lower Limit:")
        T2_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T2_lower_hour_2")
        T2_lower_minute = st.number_input("Minutes", min_value=0, value=20, step=1, key="T2_lower_minute_2")
        st.write("Upper Limit:")
        T2_upper_hour = st.number_input("Hours", min_value=0, value=6, step=1, key="T2_upper_hour_2")
        T2_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T2_upper_minute_2")

    # Input for K1 and K2
    st.subheader("Input Value Ranges for K1 and K2")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### K1")
        K1_lower = st.number_input("Lower Limit", min_value=0.0, value=0.5, step=0.001, format="%.2f", key="K1_lower_2")
        K1_upper = st.number_input("Upper Limit", min_value=0.0, value=3.0, step=0.001, format="%.2f", key="K1_upper_2")

    with col2:
        st.markdown("### K2")
        K2_lower = st.number_input("Lower Limit", min_value=0.0, value=1.0, step=0.001, format="%.2f", key="K2_lower_2")
        K2_upper = st.number_input("Upper Limit", min_value=0.0, value=7.0, step=0.001, format="%.2f", key="K2_upper_2")

    # Calculate time ranges in seconds
    T1_lower_limit = (T1_lower_hour * 60 + T1_lower_minute) * 60
    T1_upper_limit = (T1_upper_hour * 60 + T1_upper_minute) * 60
    T2_lower_limit = (T2_lower_hour * 60 + T2_lower_minute) * 60
    T2_upper_limit = (T2_upper_hour * 60 + T2_upper_minute) * 60

    # Generate allowed values
    allowed_T1_values = [x * 60 for x in range(T1_lower_limit // 60, T1_upper_limit // 60 + 1, 10)]
    allowed_T2_values = [x * 60 for x in range(T2_lower_limit // 60, T2_upper_limit // 60 + 1, 10)]
    allowed_K1_values = [round(x, 3) for x in np.arange(K1_lower, K1_upper + 0.001, 0.001)]
    allowed_K2_values = [round(x, 3) for x in np.arange(K2_lower, K2_upper + 0.001, 0.001)]

    # st.write("Allowed T1 Values (seconds):", allowed_T1_values)
    # st.write("Allowed T2 Values (seconds):", allowed_T2_values)
    # st.write("Allowed K1 Values:", allowed_K1_values)
    # st.write("Allowed K2 Values:", allowed_K2_values)

elif num_unit_hydrographs == 3:
    st.subheader("Input Time Ranges for T1, T2, and T3")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### T1")
        st.write("Lower Limit:")
        T1_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T1_lower_hour_3")
        T1_lower_minute = st.number_input("Minutes", min_value=0, value=10, step=1, key="T1_lower_minute_3")
        st.write("Upper Limit:")
        T1_upper_hour = st.number_input("Hours", min_value=0, value=2, step=1, key="T1_upper_hour_3")
        T1_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T1_upper_minute_3")

    with col2:
        st.markdown("### T2")
        st.write("Lower Limit:")
        T2_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T2_lower_hour_3")
        T2_lower_minute = st.number_input("Minutes", min_value=0, value=20, step=1, key="T2_lower_minute_3")
        st.write("Upper Limit:")
        T2_upper_hour = st.number_input("Hours", min_value=0, value=6, step=1, key="T2_upper_hour_3")
        T2_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T2_upper_minute_3")

    with col3:
        st.markdown("### T3")
        st.write("Lower Limit:")
        T3_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T3_lower_hour_3")
        T3_lower_minute = st.number_input("Minutes", min_value=0, value=20, step=1, key="T3_lower_minute_3")
        st.write("Upper Limit:")
        T3_upper_hour = st.number_input("Hours", min_value=0, value=6, step=1, key="T3_upper_hour_3")
        T3_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T3_upper_minute_3")    

    # Input for K1, K2 and K3
    st.subheader("Input Value Ranges for K1, K2 and K3")

    col1, col2, col3  = st.columns(3)

    with col1:
        st.markdown("### K1")
        K1_lower = st.number_input("Lower Limit", min_value=0.0, value=0.5, step=0.001, format="%.2f", key="K1_lower_3")
        K1_upper = st.number_input("Upper Limit", min_value=0.0, value=3.0, step=0.001, format="%.2f", key="K1_upper_3")

    with col2:
        st.markdown("### K2")
        K2_lower = st.number_input("Lower Limit", min_value=0.0, value=1.0, step=0.001, format="%.2f", key="K2_lower_3")
        K2_upper = st.number_input("Upper Limit", min_value=0.0, value=7.0, step=0.001, format="%.2f", key="K2_upper_3")
    
    with col3:
        st.markdown("### K3")
        K3_lower = st.number_input("Lower Limit", min_value=0.0, value=1.0, step=0.001, format="%.2f", key="K3_lower_3")
        K3_upper = st.number_input("Upper Limit", min_value=0.0, value=7.0, step=0.001, format="%.2f", key="K3_upper_3")
    
    # Calculate time ranges in seconds
    T1_lower_limit = (T1_lower_hour * 60 + T1_lower_minute) * 60
    T1_upper_limit = (T1_upper_hour * 60 + T1_upper_minute) * 60
    T2_lower_limit = (T2_lower_hour * 60 + T2_lower_minute) * 60
    T2_upper_limit = (T2_upper_hour * 60 + T2_upper_minute) * 60
    T3_lower_limit = (T3_lower_hour * 60 + T3_lower_minute) * 60
    T3_upper_limit = (T3_upper_hour * 60 + T3_upper_minute) * 60

    # Generate allowed values
    allowed_T1_values = [x * 60 for x in range(T1_lower_limit // 60, T1_upper_limit // 60 + 1, 10)]
    allowed_T2_values = [x * 60 for x in range(T2_lower_limit // 60, T2_upper_limit // 60 + 1, 10)]
    allowed_T3_values = [x * 60 for x in range(T3_lower_limit // 60, T3_upper_limit // 60 + 1, 10)]
    allowed_K1_values = [round(x, 3) for x in np.arange(K1_lower, K1_upper + 0.001, 0.001)]
    allowed_K2_values = [round(x, 3) for x in np.arange(K2_lower, K2_upper + 0.001, 0.001)]
    allowed_K3_values = [round(x, 3) for x in np.arange(K3_lower, K3_upper + 0.001, 0.001)]

    # st.write("Allowed T1 Values (seconds):", allowed_T1_values)
    # st.write("Allowed T2 Values (seconds):", allowed_T2_values)
    # st.write("Allowed K1 Values:", allowed_K1_values)
    # st.write("Allowed K2 Values:", allowed_K2_values)


    # (Similar code structure for T1, T2, T3 and K1, K2, K3 as shown in the sample above)

elif num_unit_hydrographs == 4:
    st.subheader("Input Time Ranges for T1, T2, T3, and T4")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### T1")
        st.write("Lower Limit:")
        T1_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T1_lower_hour_4")
        T1_lower_minute = st.number_input("Minutes", min_value=0, value=10, step=1, key="T1_lower_minute_4")
        st.write("Upper Limit:")
        T1_upper_hour = st.number_input("Hours", min_value=0, value=2, step=1, key="T1_upper_hour_4")
        T1_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T1_upper_minute_4")

    with col2:
        st.markdown("### T2")
        st.write("Lower Limit:")
        T2_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T2_lower_hour_4")
        T2_lower_minute = st.number_input("Minutes", min_value=0, value=20, step=1, key="T2_lower_minute_4")
        st.write("Upper Limit:")
        T2_upper_hour = st.number_input("Hours", min_value=0, value=6, step=1, key="T2_upper_hour_4")
        T2_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T2_upper_minute_4")

    with col3:
        st.markdown("### T3")
        st.write("Lower Limit:")
        T3_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T3_lower_hour_4")
        T3_lower_minute = st.number_input("Minutes", min_value=0, value=20, step=1, key="T3_lower_minute_4")
        st.write("Upper Limit:")
        T3_upper_hour = st.number_input("Hours", min_value=0, value=6, step=1, key="T3_upper_hour_4")
        T3_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T3_upper_minute_4") 
        
    with col4:
        st.markdown("### T4")
        st.write("Lower Limit:")
        T4_lower_hour = st.number_input("Hours", min_value=0, value=0, step=1, key="T4_lower_hour_4")
        T4_lower_minute = st.number_input("Minutes", min_value=0, value=40, step=1, key="T4_lower_minute_4")
        st.write("Upper Limit:")
        T4_upper_hour = st.number_input("Hours", min_value=0, value=10, step=1, key="T4_upper_hour_4")
        T4_upper_minute = st.number_input("Minutes", min_value=0, value=0, step=1, key="T4_upper_minute_4")         

    # Input for K1, K2 and K3
    st.subheader("Input Value Ranges for K1, K2 and K3")

    col1, col2, col3, col4  = st.columns(4)

    with col1:
        st.markdown("### K1")
        K1_lower = st.number_input("Lower Limit", min_value=0.0, value=0.5, step=0.001, format="%.2f", key="K1_lower_4")
        K1_upper = st.number_input("Upper Limit", min_value=0.0, value=3.0, step=0.001, format="%.2f", key="K1_upper_4")

    with col2:
        st.markdown("### K2")
        K2_lower = st.number_input("Lower Limit", min_value=0.0, value=1.0, step=0.001, format="%.2f", key="K2_lower_4")
        K2_upper = st.number_input("Upper Limit", min_value=0.0, value=5.0, step=0.001, format="%.2f", key="K2_upper_4")
    
    with col3:
        st.markdown("### K3")
        K3_lower = st.number_input("Lower Limit", min_value=0.0, value=3.0, step=0.001, format="%.2f", key="K3_lower_4")
        K3_upper = st.number_input("Upper Limit", min_value=0.0, value=7.0, step=0.001, format="%.2f", key="K3_upper_4")
    
    with col4:
        st.markdown("### K4")
        K4_lower = st.number_input("Lower Limit", min_value=0.0, value=4.0, step=0.001, format="%.2f", key="K4_lower_4")
        K4_upper = st.number_input("Upper Limit", min_value=0.0, value=10.0, step=0.001, format="%.2f", key="K4_upper_4")

    # Calculate time ranges in seconds
    T1_lower_limit = (T1_lower_hour * 60 + T1_lower_minute) * 60
    T1_upper_limit = (T1_upper_hour * 60 + T1_upper_minute) * 60
    T2_lower_limit = (T2_lower_hour * 60 + T2_lower_minute) * 60
    T2_upper_limit = (T2_upper_hour * 60 + T2_upper_minute) * 60
    T3_lower_limit = (T3_lower_hour * 60 + T3_lower_minute) * 60
    T3_upper_limit = (T3_upper_hour * 60 + T3_upper_minute) * 60
    T4_lower_limit = (T4_lower_hour * 60 + T4_lower_minute) * 60
    T4_upper_limit = (T4_upper_hour * 60 + T4_upper_minute) * 60

    # Generate allowed values
    allowed_T1_values = [x * 60 for x in range(T1_lower_limit // 60, T1_upper_limit // 60 + 1, 10)]
    allowed_T2_values = [x * 60 for x in range(T2_lower_limit // 60, T2_upper_limit // 60 + 1, 10)]
    allowed_T3_values = [x * 60 for x in range(T3_lower_limit // 60, T3_upper_limit // 60 + 1, 10)]
    allowed_T4_values = [x * 60 for x in range(T4_lower_limit // 60, T4_upper_limit // 60 + 1, 10)]
    allowed_K1_values = [round(x, 3) for x in np.arange(K1_lower, K1_upper + 0.001, 0.001)]
    allowed_K2_values = [round(x, 3) for x in np.arange(K2_lower, K2_upper + 0.001, 0.001)]
    allowed_K3_values = [round(x, 3) for x in np.arange(K3_lower, K3_upper + 0.001, 0.001)]
    allowed_K4_values = [round(x, 3) for x in np.arange(K4_lower, K4_upper + 0.001, 0.001)]

## CMA ES 
def run_cma_es_3_uh():
    """Perform optimization using CMA-ES and display results."""
    def objective_function_cmaes(params_flat,obs_rdii,rainfall, delta_t, area_acres):
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx = params_flat
        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        T3 = map_index_to_value(T3_idx, allowed_T3_values)

        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
        K3 = map_index_to_value(K3_idx, allowed_K3_values)
        params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]
        sim_rdii = RDII_calculation(params, delta_t, rainfall, area_acres)

        sim_rdii = np.array(sim_rdii)

        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')
        
            # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Apply penalties for constraint violations
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2 + R3 >= Ro:
            penalty += 1000 * (R1 + R2 + R3 - Ro)
        if not (T1 < T2 < T3):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3):
            penalty += 1000

        return fitness_value + penalty
    
    # Initial guess for [R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx]
    initial_guess = [
        0.06, 2.0, 0.0,  # R1, T1_idx, K1_idx
        0.01, 4, 0.0,  # R2, T2_idx, K2_idx
        0.02, 0.0, 0.0   # R3, T3_idx, K3_idx
    ]

    # Define standard deviations for each parameter
    # R: Small values, T_idx: Larger range, K_idx: Medium range
    stds = [0.0001, 0.25, 5.0,  # For R1, T1_idx, K1_idx
            0.0001, 0.25, 4.0,  # For R2, T2_idx, K2_idx
            0.0001, 0.25, 5.0]  # For R3, T3_idx, K3_idx

    # Sigma for parameters (adjusted for scaled indices)
    sigma = 0.01

    # Define bounds for R (continuous), T_idx, and K_idx
    bounds_lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds_upper = [1.0, len(allowed_T1_values) - 1, len(allowed_K1_values) - 1,
                    1.0, len(allowed_T2_values) - 1, len(allowed_K2_values) - 1,
                    1.0, len(allowed_T3_values) - 1, len(allowed_K3_values) - 1]


    es = cma.CMAEvolutionStrategy(
        initial_guess,
        sigma,
        {
            'CMA_stds': stds,  # Per-parameter standard deviations
            'bounds': [bounds_lower, bounds_upper],  # Enforce bounds for indices
            'popsize': 10,
            'maxiter': 300
        }
    )

    es.optimize(objective_function_cmaes, args=(obs_rdii, rainfall, delta_t, area_acres))

    best_parameters = es.result.xbest
    best_objective_value = es.result.fbest

    # Extract and map indices
    R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx = best_parameters
    T1 = map_index_to_value(T1_idx, allowed_T1_values)
    T2 = map_index_to_value(T2_idx, allowed_T2_values)
    T3 = map_index_to_value(T3_idx, allowed_T3_values)
    K1 = map_index_to_value(K1_idx, allowed_K1_values)
    K2 = map_index_to_value(K2_idx, allowed_K2_values)
    K3 = map_index_to_value(K3_idx, allowed_K3_values)
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]


    # Calculate criteria
    criteria = calculate_criteria(best_params_actual, obs_rdii, delta_t, rainfall, area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]


        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_objective_value}
            </div>
            """,
            unsafe_allow_html=True
        ) 
        

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

    #with col3:
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)

def run_cma_es_2_uh():
    def objective_function_cmaes_2uh(params_flat, obs_rdii,rainfall, delta_t, area_acres):
        """
        Objective function for optimization, calculates fitness and applies penalties.

        Args:
            params_flat (list): Flattened list of parameters (R, T, K indices).
            obs_rdii (array-like): Observed RDII values.

        Returns:
            float: Objective function value (fitness + penalties).
        """
        # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx = params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        
        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
                
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2)]

        # Calculate simulated RDII
        sim_rdii = RDII_calculation_2uh(params, delta_t, rainfall, area_acres)

        # Convert to numpy array to ensure compatibility
        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Apply penalties for constraint violations
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2  >= Ro:
            penalty += 1000 * (R1 + R2  - Ro)
        if not (T1 < T2 ):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 ):
            penalty += 1000

        return fitness_value + penalty   

        # Initial guess for [R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx]
    initial_guess = [
        0.06, 2.0, 0.0,  # R1, T1_idx, K1_idx
        0.01, 4, 0.0,  # R2, T2_idx, K2_idx
    ]

    # Define standard deviations for each parameter
    # R: Small values, T_idx: Larger range, K_idx: Medium range
    stds = [0.0001, 0.25, 5.0,  # For R1, T1_idx, K1_idx
            0.0001, 0.25, 4.0  # For R2, T2_idx, K2_idx
            ]  

    # Sigma for parameters (adjusted for scaled indices)
    sigma = 0.01

    # Define bounds for R (continuous), T_idx, and K_idx
    bounds_lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds_upper = [1.0, len(allowed_T1_values) - 1, len(allowed_K1_values) - 1,
                    1.0, len(allowed_T2_values) - 1, len(allowed_K2_values) - 1]
    
    es = cma.CMAEvolutionStrategy(
        initial_guess,
        sigma,
        {
            'CMA_stds': stds,  # Per-parameter standard deviations
            'bounds': [bounds_lower, bounds_upper],  # Enforce bounds for indices
            'popsize': 10,
            'maxiter': 300
        }
    )

    es.optimize(objective_function_cmaes_2uh, args=(obs_rdii, rainfall, delta_t, area_acres))

    best_parameters = es.result.xbest
    best_objective_value = es.result.fbest

    # Extract and map indices
    R1, T1_idx, K1_idx, R2, T2_idx, K2_idx = best_parameters
    T1 = map_index_to_value(T1_idx, allowed_T1_values)
    T2 = map_index_to_value(T2_idx, allowed_T2_values)
    K1 = map_index_to_value(K1_idx, allowed_K1_values)
    K2 = map_index_to_value(K2_idx, allowed_K2_values)
    
    best_params_actual = [(R1, T1, K1), (R2, T2, K2)]


    # Calculate criteria
    criteria = calculate_criteria(best_params_actual, obs_rdii, delta_t, rainfall, area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]


        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_objective_value}
            </div>
            """,
            unsafe_allow_html=True
        ) 
        

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_2uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

    #with col3:
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)

def run_cma_es_4_uh():
    def objective_function_cmaes_4uh(params_flat, obs_rdii,rainfall, delta_t, area_acres):
        """
        Objective function for optimization, calculates fitness and applies penalties.

        Args:
            params_flat (list): Flattened list of parameters (R, T, K indices).
            obs_rdii (array-like): Observed RDII values.

        Returns:
            float: Objective function value (fitness + penalties).
        """
        # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx, R4, T4_idx, K4_idx = params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        T3 = map_index_to_value(T3_idx, allowed_T3_values)
        T4 = map_index_to_value(T4_idx, allowed_T4_values)

        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
        K3 = map_index_to_value(K3_idx, allowed_K3_values)
        K4 = map_index_to_value(K4_idx, allowed_K4_values)
        
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3), (R4, T4, K4)]

        # Calculate simulated RDII
        sim_rdii = RDII_calculation_4uh(params, delta_t, rainfall, area_acres)

        # Convert to numpy array to ensure compatibility
        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Apply penalties for constraint violations
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2 + R3+ R4 >= Ro:
            penalty += 1000 * (R1 + R2 + R3+ R4 - Ro)
        if not (T1 < T2 < T3 < T4):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3 < T4 * K4):
            penalty += 1000

        return fitness_value + penalty
    
    # Initial guess for [R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx]
    initial_guess = [
        0.06, 2.0, 0.0,  # R1, T1_idx, K1_idx
        0.01, 4, 0.0,  # R2, T2_idx, K2_idx
        0.00, 0.0, 0.0,   # R3, T3_idx, K3_idx
        0.01, 0.0, 0.0   # R4, T4_idx, K4_idx
    ]

    # Define standard deviations for each parameter
    # R: Small values, T_idx: Larger range, K_idx: Medium range
    stds = [0.0001, 0.25, 5.0,  # For R1, T1_idx, K1_idx
            0.0001, 0.25, 4.0,  # For R2, T2_idx, K2_idx
            0.0001, 0.25, 5.0,
            0.0001,0.25,5.0]  # For R3, T3_idx, K3_idx

    # Sigma for parameters (adjusted for scaled indices)
    sigma = 0.01

    # Define bounds for R (continuous), T_idx, and K_idx
    bounds_lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0]
    bounds_upper = [1.0, len(allowed_T1_values) - 1, len(allowed_K1_values) - 1,
                    1.0, len(allowed_T2_values) - 1, len(allowed_K2_values) - 1,
                    1.0, len(allowed_T3_values) - 1, len(allowed_K3_values) - 1,
                    1.0, len(allowed_T4_values) - 1, len(allowed_K4_values) - 1]
    
    es = cma.CMAEvolutionStrategy(
        initial_guess,
        sigma,
        {
            'CMA_stds': stds,  # Per-parameter standard deviations
            'bounds': [bounds_lower, bounds_upper],  # Enforce bounds for indices
            'popsize': 10,
            'maxiter': 300
        }
    )

    es.optimize(objective_function_cmaes_4uh, args=(obs_rdii, rainfall, delta_t, area_acres))

    best_parameters = es.result.xbest
    best_objective_value = es.result.fbest
    # Extract indices for T and K from the best parameters
    R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx , R4, T4_idx, K4_idx = best_parameters

    # Map indices to discrete values
    T1 = map_index_to_value(T1_idx, allowed_T1_values)
    T2 = map_index_to_value(T2_idx, allowed_T2_values)
    T3 = map_index_to_value(T3_idx, allowed_T3_values)
    T4 = map_index_to_value(T4_idx, allowed_T4_values)

    K1 = map_index_to_value(K1_idx, allowed_K1_values)
    K2 = map_index_to_value(K2_idx, allowed_K2_values)
    K3 = map_index_to_value(K3_idx, allowed_K3_values)
    K4 = map_index_to_value(K4_idx, allowed_K4_values)

    # Final best parameter set
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3) , (R4, T4, K4)]

    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2","Unit Hydrograph 3","Unit Hydrograph 4"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]


        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_objective_value}
            </div>
            """,
            unsafe_allow_html=True
        ) 
        

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_4uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)

## DE
def run_de_3_uh():
    '''perform optimization using Differential Evolution and display result '''
    def objective_function_de(params_flat,obs_rdii,rainfall, delta_t, area_acres):
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx = params_flat
        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        T3 = map_index_to_value(T3_idx, allowed_T3_values)

        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
        K3 = map_index_to_value(K3_idx, allowed_K3_values)
        params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]
        sim_rdii = RDII_calculation(params, delta_t, rainfall, area_acres)

        sim_rdii = np.array(sim_rdii)

        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')
        
            # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Apply penalties for constraint violations
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2 + R3 >= Ro:
            penalty += 1000 * (R1 + R2 + R3 - Ro)
        if not (T1 < T2 < T3):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3):
            penalty += 1000

        return fitness_value + penalty    
    
    bounds = [
        (0.0, 1.0),  # R1
        (0.0, len(allowed_T1_values) - 1),  # T1_idx
        (0.0, len(allowed_K1_values) - 1),  # K1_idx
        (0.0, 1.0),  # R2
        (0.0, len(allowed_T2_values) - 1),  # T2_idx
        (0.0, len(allowed_K2_values) - 1),  # K2_idx
        (0.0, 1.0),  # R3
        (0.0, len(allowed_T3_values) - 1),  # T3_idx
        (0.0, len(allowed_K3_values) - 1)   # K3_idx
    ]
    result = differential_evolution(
        objective_function_de,
        bounds,
        args=(obs_rdii, rainfall, delta_t, area_acres),  
        strategy='best1bin',
        maxiter=300,
        popsize=30,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7
    )

    # Extract the best parameters
    best_parameters = result.x
    best_fitness_value = result.fun

    # Map indices back to actual values
    R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx = best_parameters

    T1 = map_index_to_value(T1_idx, allowed_T1_values)
    T2 = map_index_to_value(T2_idx, allowed_T2_values)
    T3 = map_index_to_value(T3_idx, allowed_T3_values)

    K1 = map_index_to_value(K1_idx, allowed_K1_values)
    K2 = map_index_to_value(K2_idx, allowed_K2_values)
    K3 = map_index_to_value(K3_idx, allowed_K3_values)

    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]

    criteria = calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

    #with col3:
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)   

def run_de_2_uh():
    def objective_function_de_2uh(params_flat,obs_rdii,rainfall, delta_t, area_acres):
        """
        Objective function for optimization, calculates fitness and applies penalties.

        Args:
            params_flat (list): Flattened list of parameters (R, T, K indices).
            obs_rdii (array-like): Observed RDII values.

        Returns:
            float: Objective function value (fitness + penalties).
        """
        # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx = params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        
        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
                
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2)]

        # Calculate simulated RDII
        sim_rdii = RDII_calculation_2uh(params, delta_t, rainfall, area_acres)

        # Convert to numpy array to ensure compatibility
        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Apply penalties for constraint violations
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2  >= Ro:
            penalty += 1000 * (R1 + R2  - Ro)
        if not (T1 < T2 ):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 ):
            penalty += 1000

        return fitness_value + penalty  

    bounds = [
        (0.0, 1.0),  # R1
        (0.0, len(allowed_T1_values) - 1),  # T1_idx
        (0.0, len(allowed_K1_values) - 1),  # K1_idx
        (0.0, 1.0),  # R2
        (0.0, len(allowed_T2_values) - 1),  # T2_idx
        (0.0, len(allowed_K2_values) - 1),  # K2_idx
    ]

    result = differential_evolution(
        objective_function_de_2uh,
        bounds,
        args=(obs_rdii, rainfall, delta_t, area_acres),  # Pass obs_rdii as an argument
        strategy='best1bin',
        maxiter=300,
        popsize=30,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7
    )

    # Extract the best parameters
    best_parameters = result.x
    best_fitness_value = result.fun
    # Map indices back to actual values
    R1, T1_idx, K1_idx, R2, T2_idx, K2_idx = best_parameters

    T1 = map_index_to_value(T1_idx, allowed_T1_values)
    T2 = map_index_to_value(T2_idx, allowed_T2_values)

    K1 = map_index_to_value(K1_idx, allowed_K1_values)
    K2 = map_index_to_value(K2_idx, allowed_K2_values)
    best_params_actual = [(R1, T1, K1), (R2, T2, K2)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_2uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)

def run_de_4_uh():
    def objective_function_de_4uh(params_flat,obs_rdii,rainfall, delta_t, area_acres):
       # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx, R4, T4_idx, K4_idx = params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        T3 = map_index_to_value(T3_idx, allowed_T3_values)
        T4 = map_index_to_value(T4_idx, allowed_T4_values)

        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
        K3 = map_index_to_value(K3_idx, allowed_K3_values)
        K4 = map_index_to_value(K4_idx, allowed_K4_values)
        
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3), (R4, T4, K4)]

            # Calculate simulated RDII
        sim_rdii = RDII_calculation_4uh(params, delta_t, rainfall, area_acres)

        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')
        
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Constraints and penalties
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2 + R3 + R4 >= Ro:
            penalty += 1000 * (R1 + R2 + R3 + R4 - Ro)
        if not (T1 < T2 < T3 < T4):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3 < T4 * K4):
            penalty += 1000

        return fitness_value + penalty 
    
    bounds = [
        (0.0, 1.0),  # R1
        (0.0, len(allowed_T1_values) - 1),  # T1_idx
        (0.0, len(allowed_K1_values) - 1),  # K1_idx
        (0.0, 1.0),  # R2
        (0.0, len(allowed_T2_values) - 1),  # T2_idx
        (0.0, len(allowed_K2_values) - 1),  # K2_idx
        (0.0, 1.0),  # R3
        (0.0, len(allowed_T3_values) - 1),  # T3_idx
        (0.0, len(allowed_K3_values) - 1),   # K3_idx
        (0.0, 1.0),  # R4
        (0.0, len(allowed_T4_values) - 1),  # T4_idx
        (0.0, len(allowed_K4_values) - 1)   # K4_idx    
    ]
    result = differential_evolution(
        objective_function_de_4uh,
        bounds,
        args=(obs_rdii, rainfall, delta_t, area_acres),  # Pass obs_rdii as an argument
        strategy='best1bin',
        maxiter=300,
        popsize=30,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7
    )

    # Extract the best parameters
    best_parameters = result.x
    best_fitness_value = result.fun
    # Map indices back to actual values
    R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx, R4, T4_idx, K4_idx  = best_parameters

    T1 = map_index_to_value(T1_idx, allowed_T1_values)
    T2 = map_index_to_value(T2_idx, allowed_T2_values)
    T3 = map_index_to_value(T3_idx, allowed_T3_values)
    T4 = map_index_to_value(T4_idx, allowed_T4_values)

    K1 = map_index_to_value(K1_idx, allowed_K1_values)
    K2 = map_index_to_value(K2_idx, allowed_K2_values)
    K3 = map_index_to_value(K3_idx, allowed_K3_values)
    K4 = map_index_to_value(K4_idx, allowed_K4_values)
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3), (R4, T4, K4)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3", "Unit Hydrograph 4"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_4uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)


## GA 
def run_ga_3_uh():
    """perform optimization using Genetic Algorithm and display result"""

    # Define the problem and fitness function
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the objective function

    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Define genes (parameters)
    toolbox.register("R", random.uniform, 0, 1)  # R values
    toolbox.register("T1_index", random.randint, 0, len(allowed_T1_values) - 1)  # Index for T1
    toolbox.register("T2_index", random.randint, 0, len(allowed_T2_values) - 1)  # Index for T2
    toolbox.register("T3_index", random.randint, 0, len(allowed_T3_values) - 1)  # Index for T3
    toolbox.register("K1_index", random.randint, 0, len(allowed_K1_values) - 1)  # Index for K1
    toolbox.register("K2_index", random.randint, 0, len(allowed_K2_values) - 1)  # Index for K2
    toolbox.register("K3_index", random.randint, 0, len(allowed_K3_values) - 1)  # Index for K3

    # Define individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.R, toolbox.T1_index, toolbox.K1_index,
                    toolbox.R, toolbox.T2_index, toolbox.K2_index,
                    toolbox.R, toolbox.T3_index, toolbox.K3_index),
                    n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Register the population function    

    # Register evaluation, mutation, crossover, and selection
    toolbox.register("evaluate", objective_function)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Gaussian mutation
    #toolbox.register("select", tools.selBest)
    toolbox.register("select", tools.selTournament, tournsize=3)   

    # Run the Genetic Algorithm
    population = toolbox.population(n=100)  # Population size
    result, log = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.5,  # Crossover probability
        mutpb=0.20,  # Mutation probability
        ngen=300,  # Number of generations
        verbose=True
    )

    # Extract the best solution
    best_individual = tools.selBest(population, k=1)[0]
    best_params = best_individual
    R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index = best_params

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    T3 = allowed_T3_values[int(T3_index)]

    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    K3 = allowed_K3_values[int(K3_index)]

    # Extract the best fitness function value
    best_fitness_value = best_individual.fitness.values[0]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)
    
    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

    #with col3:
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)       

def run_ga_2_uh():
    def objective_function_ga_2uh(params):
        """
        Compute RMSE + penalty for given parameters.
        """
        # Convert flat array back to parameter tuples
        R1, T1_index, K1_index, R2, T2_index, K2_index = params

        # Map indices to actual T and K values
        T1 = allowed_T1_values[int(T1_index)]
        T2 = allowed_T2_values[int(T2_index)]
        
        K1 = allowed_K1_values[int(K1_index)]
        K2 = allowed_K2_values[int(K2_index)]
        
        # Package parameters
        param_tuples = [(R1, T1, K1), (R2, T2, K2)]

        # Calculate simulated RDII
        sim_rdii = RDII_calculation_2uh(param_tuples, delta_t, rainfall, area_acres)

        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

                
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        # Calculate penalties
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t,area_acres)

        # R constraints
        if R1 + R2 > Ro:
            penalty += 1000 * (R1 + R2 - Ro)

        # T and K constraints
        if not (T1 < T2):
            penalty += 1000
        if not (T1 * K1 < T2 * K2):
            penalty += 1000

        return (fitness_value + penalty,)  # Ensure the result is a tuple for GA algorithms
    
    # Define the problem and fitness function
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the objective function

    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Define genes (parameters)
    toolbox.register("R", random.uniform, 0, 1)  # R values
    toolbox.register("T1_index", random.randint, 0, len(allowed_T1_values) - 1)  # Index for T1
    toolbox.register("T2_index", random.randint, 0, len(allowed_T2_values) - 1)  # Index for T2
    #toolbox.register("T3_index", random.randint, 0, len(allowed_T3_values) - 1)  # Index for T3
    toolbox.register("K1_index", random.randint, 0, len(allowed_K1_values) - 1)  # Index for K1
    toolbox.register("K2_index", random.randint, 0, len(allowed_K2_values) - 1)  # Index for K2
    #toolbox.register("K3_index", random.randint, 0, len(allowed_K3_values) - 1)  # Index for K3

    # Define individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.R, toolbox.T1_index, toolbox.K1_index,
                    toolbox.R, toolbox.T2_index, toolbox.K2_index),
                    #toolbox.R, toolbox.T3_index, toolbox.K3_index),
                    n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Register the population function

    # Register evaluation, mutation, crossover, and selection
    toolbox.register("evaluate", objective_function_ga_2uh)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Gaussian mutation
    #toolbox.register("select", tools.selBest)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the Genetic Algorithm
    population = toolbox.population(n=100)  # Population size
    result, log = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.5,  # Crossover probability
        mutpb=0.20,  # Mutation probability
        ngen=100,  # Number of generations
        verbose=True
    )

    # Extract the best solution
    best_individual = tools.selBest(population, k=1)[0]
    best_params = best_individual
    R1, T1_index, K1_index, R2, T2_index, K2_index = best_params

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    
    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    best_fitness_value = best_individual.fitness.values[0]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2)]  
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_2uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)

def run_ga_4_uh():
    def objective_function_ga_4uh(params):
        """
        Compute RMSE + penalty for given parameters.
        """
        # Convert flat array back to parameter tuples
        R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index , R4, T4_index, K4_index = params

        # Map indices to actual T and K values
        T1 = allowed_T1_values[int(T1_index)]
        T2 = allowed_T2_values[int(T2_index)]
        T3 = allowed_T3_values[int(T3_index)]
        T4 = allowed_T4_values[int(T4_index)]

        K1 = allowed_K1_values[int(K1_index)]
        K2 = allowed_K2_values[int(K2_index)]
        K3 = allowed_K3_values[int(K3_index)]
        K4 = allowed_K4_values[int(K4_index)]

        # Package parameters
        param_tuples = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3) , (R4, T4, K4)]

        # Calculate simulated RDII
        sim_rdii = RDII_calculation_4uh(param_tuples, delta_t, rainfall, area_acres)

        sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        # # Calculate fitness (RMSE)
        # rmse = fitness(obs_rdii, sim_rdii, delta_t)
        
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )


        # Calculate penalties
        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t,area_acres)

        # R constraints
        if R1 + R2 + R3 +R4 > Ro:
            penalty += 1000 * (R1 + R2 + R3 +R4 - Ro)

        # T and K constraints
        if not (T1 < T2 < T3 < T4):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3 < T4 * K4):
            penalty += 1000

        return (fitness_value + penalty,)  # Ensure the result is a tuple for GA algorithms 

    # Define the problem and fitness function
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the objective function

    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Define genes (parameters)
    toolbox.register("R", random.uniform, 0, 1)  # R values
    toolbox.register("T1_index", random.randint, 0, len(allowed_T1_values) - 1)  # Index for T1
    toolbox.register("T2_index", random.randint, 0, len(allowed_T2_values) - 1)  # Index for T2
    toolbox.register("T3_index", random.randint, 0, len(allowed_T3_values) - 1)  # Index for T3
    toolbox.register("T4_index", random.randint, 0, len(allowed_T4_values) - 1)  # Index for T3
    toolbox.register("K1_index", random.randint, 0, len(allowed_K1_values) - 1)  # Index for K1
    toolbox.register("K2_index", random.randint, 0, len(allowed_K2_values) - 1)  # Index for K2
    toolbox.register("K3_index", random.randint, 0, len(allowed_K3_values) - 1)  # Index for K3
    toolbox.register("K4_index", random.randint, 0, len(allowed_K4_values) - 1)  # Index for K3

    # Define individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.R, toolbox.T1_index, toolbox.K1_index,
                    toolbox.R, toolbox.T2_index, toolbox.K2_index,
                    toolbox.R, toolbox.T3_index, toolbox.K3_index,
                    toolbox.R, toolbox.T4_index, toolbox.K4_index),
                    n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Register the population function

    # Register evaluation, mutation, crossover, and selection
    toolbox.register("evaluate", objective_function_ga_4uh)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Gaussian mutation
    #toolbox.register("select", tools.selBest)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Run the Genetic Algorithm
    population = toolbox.population(n=100)  # Population size
    result, log = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.5,  # Crossover probability
        mutpb=0.20,  # Mutation probability
        ngen=300,  # Number of generations
        verbose=True
    )

    # Extract the best solution
    best_individual = tools.selBest(population, k=1)[0]
    best_params = best_individual
    R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index,  R4, T4_index, K4_index  = best_params

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    T3 = allowed_T3_values[int(T3_index)]
    T4 = allowed_T4_values[int(T4_index)]

    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    K3 = allowed_K3_values[int(K3_index)]
    K4 = allowed_K4_values[int(K4_index)]

    # Extract the best fitness function value
    best_fitness_value = best_individual.fitness.values[0]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3) , (R4, T4, K4)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    # Prepare criteria data for tabular display
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    })

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3", "Unit Hydrograph 4"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_4uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)

## SA

def run_sa_3_uh():
    '''perform Simulated Annealing optimization and display result'''
    def objective_function_sa(params_flat):
        # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx = params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        T3 = map_index_to_value(T3_idx, allowed_T3_values)

        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
        K3 = map_index_to_value(K3_idx, allowed_K3_values)
        
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]

        # Proceed with sim_rdii, RMSE, and penalties as before
        sim_rdii = RDII_calculation(params, delta_t, rainfall, area_acres)

        # if not sim_rdii or len(sim_rdii) == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
        #     return float('inf')
        # if len(sim_rdii) != len(obs_rdii):
        #     sim_rdii = sim_rdii[:len(obs_rdii)]

        #     sim_rdii = np.array(sim_rdii)

        # Check for invalid outputs
        sim_rdii = np.array(sim_rdii)
        
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')


        
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )


        # rmse = fitness(obs_rdii, sim_rdii, delta_t)

    

        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2 + R3 >= Ro:
            penalty += 1000 * (R1 + R2 + R3 - Ro)
        if not (T1 < T2 < T3):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3):
            penalty += 1000

        return fitness_value + penalty


    # Define bounds for all parameters
    bounds = [
        (0, 1),  # R1
        (0, len(allowed_T1_values) - 1),  # T1 index
        (0, len(allowed_K1_values) - 1),  # K1 index
        (0, 1),  # R2
        (0, len(allowed_T2_values) - 1),  # T2 index
        (0, len(allowed_K2_values) - 1),  # K2 index
        (0, 1),  # R3
        (0, len(allowed_T3_values) - 1),  # T3 index
        (0, len(allowed_K3_values) - 1),  # K3 index
    ]  

    # Perform Simulated Annealing
    result = dual_annealing(
        objective_function_sa,
        bounds=bounds,
        maxiter=1000,  # Maximum number of iterations
        initial_temp=5230.0,  # Initial temperature (optional tuning)
        restart_temp_ratio=2e-5,  # Temperature restart ratio
    ) 

    # Extract the best solution
    best_params = result.x
    best_fitness_value = result.fun

    # Map indices to actual T and K values
    R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index = best_params

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    T3 = allowed_T3_values[int(T3_index)]

    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    K3 = allowed_K3_values[int(K3_index)]    
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)] 
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    }) 

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

    #with col3:
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t) 

def run_sa_2_uh():
    def objective_function_sa_2uh(params_flat):
        # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx = params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        
        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
                
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2)]

        # Proceed with sim_rdii, RMSE, and penalties as before
        sim_rdii = RDII_calculation_2uh(params, delta_t, rainfall, area_acres)
        
        # Check for invalid outputs
        sim_rdii = np.array(sim_rdii)
        
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')
        
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )

        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2  >= Ro:
            penalty += 1000 * (R1 + R2  - Ro)
        if not (T1 < T2 ):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 ):
            penalty += 1000

        return fitness_value + penalty    
    
    # Define bounds for all parameters
    bounds = [
        (0, 1),  # R1
        (0, len(allowed_T1_values) - 1),  # T1 index
        (0, len(allowed_K1_values) - 1),  # K1 index
        (0, 1),  # R2
        (0, len(allowed_T2_values) - 1),  # T2 index
        (0, len(allowed_K2_values) - 1),  # K2 index   
    ]  

    # Perform Simulated Annealing
    result = dual_annealing(
        objective_function_sa_2uh,
        bounds=bounds,
        maxiter=1000,  # Maximum number of iterations
        initial_temp=5230.0,  # Initial temperature (optional tuning)
        restart_temp_ratio=2e-5,  # Temperature restart ratio
    )

    # Extract the best solution
    best_params = result.x
    best_fitness_value = result.fun

    # Map indices to actual T and K values
    R1, T1_index, K1_index, R2, T2_index, K2_index = best_params

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
   
    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    }) 

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_2uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t) 
    
def run_sa_4_uh():
    def objective_function_sa_4uh(params_flat):
        # Unpack parameters (R remains continuous, T and K are indices)
        R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx , R4, T4_idx, K4_idx= params_flat

        # Map indices to actual discrete values
        T1 = map_index_to_value(T1_idx, allowed_T1_values)
        T2 = map_index_to_value(T2_idx, allowed_T2_values)
        T3 = map_index_to_value(T3_idx, allowed_T3_values)
        T4 = map_index_to_value(T4_idx, allowed_T4_values)

        K1 = map_index_to_value(K1_idx, allowed_K1_values)
        K2 = map_index_to_value(K2_idx, allowed_K2_values)
        K3 = map_index_to_value(K3_idx, allowed_K3_values)
        K4 = map_index_to_value(K4_idx, allowed_K4_values)
        
        # Parameters are now R (continuous) and T/K (discrete via mapping)
        params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3), (R4, T4, K4)]

        # Proceed with sim_rdii, RMSE, and penalties as before
        sim_rdii = RDII_calculation_4uh(params, delta_t, rainfall, area_acres)

        # Check for invalid outputs
        sim_rdii = np.array(sim_rdii)
        
        if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
            print("Invalid sim_rdii generated, returning large penalty.")
            return float('inf')

        # Pad the shorter array with zeros to match the length of the longer one
        max_length = max(len(obs_rdii), len(sim_rdii))
        obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
        sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')
        
        # Calculate fitness value
        fitness_value = fitness(
            obs_rdii_padded,
            sim_rdii_padded,
            delta_t,
            weight_rmse=0.25,
            weight_r2=0.25,
            weight_pbias=0.25,
            weight_nse=0.25
        )
         

        penalty = 0
        Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
        if R1 + R2 + R3 + R4 >= Ro:
            penalty += 1000 * (R1 + R2 + R3 +R4 - Ro)
        if not (T1 < T2 < T3 < T4):
            penalty += 1000
        if not (T1 * K1 < T2 * K2 < T3 * K3 < T4):
            penalty += 1000

        return fitness_value + penalty
    # Define bounds for all parameters
    bounds = [
        (0, 1),  # R1
        (0, len(allowed_T1_values) - 1),  # T1 index
        (0, len(allowed_K1_values) - 1),  # K1 index
        (0, 1),  # R2
        (0, len(allowed_T2_values) - 1),  # T2 index
        (0, len(allowed_K2_values) - 1),  # K2 index
        (0, 1),  # R3
        (0, len(allowed_T3_values) - 1),  # T3 index
        (0, len(allowed_K3_values) - 1),  # K3 index
        (0, 1),  # R4
        (0, len(allowed_T4_values) - 1),  # T4 index
        (0, len(allowed_K4_values) - 1),  # K4 index
    ] 
    # Perform Simulated Annealing
    result = dual_annealing(
        objective_function_sa_4uh,
        bounds=bounds,
        maxiter=1000,  # Maximum number of iterations
        initial_temp=5230.0,  # Initial temperature (optional tuning)
        restart_temp_ratio=2e-5,  # Temperature restart ratio
    )

    # Extract the best solution
    best_params = result.x
    best_fitness_value = result.fun

    # Map indices to actual T and K values
    R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index , R4, T4_index, K4_index = best_params

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    T3 = allowed_T3_values[int(T3_index)]
    T4 = allowed_T4_values[int(T4_index)]

    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    K3 = allowed_K3_values[int(K3_index)]
    K4 = allowed_K4_values[int(K4_index)]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3) , (R4, T4, K4)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)

    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    }) 

    # Show results
    #st.write("Best Objective Value:", best_objective_value)
    #st.write("Best Parameters (Actual Values):", best_params_actual)
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3", "Unit Hydrograph 4"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_4uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t) 


def run_pso_3_uh():
    '''perform Particle Swarm optimization and display result'''
    def objective_function_pso(params_flat):
        """
        Objective function for Particle Swarm Optimization (PSO).
        Processes an array of particle positions and computes the fitness for each particle.

        Args:
            params_flat (ndarray): Array of particle positions with shape (n_particles, dimensions).

        Returns:
            ndarray: Fitness values for all particles.
        """
        fitness_values = []  # List to store fitness for each particle

        for particle in params_flat:
            # Unpack parameters for the current particle
            R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx = particle

            # Map indices to actual discrete values
            T1 = map_index_to_value(T1_idx, allowed_T1_values)
            T2 = map_index_to_value(T2_idx, allowed_T2_values)
            T3 = map_index_to_value(T3_idx, allowed_T3_values)

            K1 = map_index_to_value(K1_idx, allowed_K1_values)
            K2 = map_index_to_value(K2_idx, allowed_K2_values)
            K3 = map_index_to_value(K3_idx, allowed_K3_values)

            # Combine parameters into a structured format
            params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]

            # Simulate RDII using the current parameters
            sim_rdii = RDII_calculation(params, delta_t, rainfall, area_acres)

            
            sim_rdii = np.array(sim_rdii)

            # Check for invalid outputs
            if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
                print("Invalid sim_rdii generated, returning large penalty.")
                return float('inf')

            # Pad the shorter array with zeros to match the length of the longer one
            max_length = max(len(obs_rdii), len(sim_rdii))
            obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
            sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')


        
            # Calculate fitness value
            fitness_value = fitness(
                obs_rdii_padded,
                sim_rdii_padded,
                delta_t,
                weight_rmse=0.25,
                weight_r2=0.25,
                weight_pbias=0.25,
                weight_nse=0.25
            )
        
            
            # Apply penalties for constraints
            penalty = 0
            Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
            if R1 + R2 + R3 >= Ro:
                penalty += 100 * (R1 + R2 + R3 - Ro)
            if not (T1 < T2 < T3):
                penalty += 100
            if not (T1 * K1 < T2 * K2 < T3 * K3):
                penalty += 100

            # Final fitness value (RMSE + penalties)
            total_fitness = fitness_value + penalty
            fitness_values.append(total_fitness)

        return np.array(fitness_values)  # Return fitness for all particles

    # Define bounds for all parameters
    bounds = [
        (0, 1),  # R1
        (0, len(allowed_T1_values) - 1),  # T1 index
        (0, len(allowed_K1_values) - 1),  # K1 index
        (0, 1),  # R2
        (0, len(allowed_T2_values) - 1),  # T2 index
        (0, len(allowed_K2_values) - 1),  # K2 index
        (0, 1),  # R3
        (0, len(allowed_T3_values) - 1),  # T3 index
        (0, len(allowed_K3_values) - 1),  # K3 index
    ]

    # Define the number of particles and iterations
    num_particles = 100
    num_iterations = 300

    # Initialize the PSO optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles,
        dimensions=len(bounds),
        options={"c1": 1.5, "c2": 1.5, "w": 0.7},
        bounds=(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])),  # Extract lower and upper bounds,
    )

    # Perform optimization
    best_cost, best_pos = optimizer.optimize(objective_function_pso, iters=num_iterations)    
    # Map indices to actual T and K values
    R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index = best_pos

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    T3 = allowed_T3_values[int(T3_index)]

    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    K3 = allowed_K3_values[int(K3_index)]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)
    best_fitness_value = best_cost
    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    }) 

    # Show results
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)

    #with col3:
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t) 

def run_pso_2_uh():
    def objective_function_pso_2uh(params_flat):
        """
        Objective function for Particle Swarm Optimization (PSO).
        Processes an array of particle positions and computes the fitness for each particle.

        Args:
            params_flat (ndarray): Array of particle positions with shape (n_particles, dimensions).

        Returns:
            ndarray: Fitness values for all particles.
        """
        fitness_values = []  # List to store fitness for each particle

        for particle in params_flat:
            # Unpack parameters for the current particle
            R1, T1_idx, K1_idx, R2, T2_idx, K2_idx = particle

            # Map indices to actual discrete values
            T1 = map_index_to_value(T1_idx, allowed_T1_values)
            T2 = map_index_to_value(T2_idx, allowed_T2_values)
            #T3 = map_index_to_value(T3_idx, allowed_T3_values)

            K1 = map_index_to_value(K1_idx, allowed_K1_values)
            K2 = map_index_to_value(K2_idx, allowed_K2_values)
            #K3 = map_index_to_value(K3_idx, allowed_K3_values)

            # Combine parameters into a structured format
            params = [(R1, T1, K1), (R2, T2, K2)]

            # Simulate RDII using the current parameters
            sim_rdii = RDII_calculation_2uh(params, delta_t, rainfall, area_acres)

            
            # rmse = fitness(obs_rdii, sim_rdii, delta_t)
            sim_rdii = np.array(sim_rdii)

            # Check for invalid outputs
            if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
                print("Invalid sim_rdii generated, returning large penalty.")
                return float('inf')

            # Pad the shorter array with zeros to match the length of the longer one
            max_length = max(len(obs_rdii), len(sim_rdii))
            obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
            sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')
        
            # Calculate fitness value
            fitness_value = fitness(
                obs_rdii_padded,
                sim_rdii_padded,
                delta_t,
                weight_rmse=0.25,
                weight_r2=0.25,
                weight_pbias=0.25,
                weight_nse=0.25
            )

            
            # Apply penalties for constraints
            penalty = 0
            Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
            if R1 + R2  >= Ro:
                penalty += 100 * (R1 + R2  - Ro)
            if not (T1 < T2 ):
                penalty += 100
            if not (T1 * K1 < T2 * K2 ):
                penalty += 100

            # Final fitness value (RMSE + penalties)
            total_fitness = fitness_value + penalty
            fitness_values.append(total_fitness)

        return np.array(fitness_values)  # Return fitness for all particles
    
    # Define bounds for all parameters
    bounds = [
        (0, 1),  # R1
        (0, len(allowed_T1_values) - 1),  # T1 index
        (0, len(allowed_K1_values) - 1),  # K1 index
        (0, 1),  # R2
        (0, len(allowed_T2_values) - 1),  # T2 index
        (0, len(allowed_K2_values) - 1),  # K2 index
        #(0, 1),  # R3
        #(0, len(allowed_T3_values) - 1),  # T3 index
        #(0, len(allowed_K3_values) - 1),  # K3 index
    ]

    # Define the number of particles and iterations
    num_particles = 100
    num_iterations = 300

    # Initialize the PSO optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles,
        dimensions=len(bounds),
        options={"c1": 1.5, "c2": 1.5, "w": 0.7},
        bounds=(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])),  # Extract lower and upper bounds,
    )

    # Perform optimization
    best_cost, best_pos = optimizer.optimize(objective_function_pso_2uh, iters=num_iterations)
    # Map indices to actual T and K values
    R1, T1_index, K1_index, R2, T2_index, K2_index = best_pos

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    
    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2)]   
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)  
    best_fitness_value = best_cost

    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    }) 

    # Show results
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_2uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)
   
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t)  

def run_pso_4_uh():
    def objective_function_pso_4uh(params_flat):
        """
        Objective function for Particle Swarm Optimization (PSO).
        Processes an array of particle positions and computes the fitness for each particle.

        Args:
            params_flat (ndarray): Array of particle positions with shape (n_particles, dimensions).

        Returns:
            ndarray: Fitness values for all particles.
        """
        fitness_values = []  # List to store fitness for each particle

        for particle in params_flat:
            # Unpack parameters for the current particle
            R1, T1_idx, K1_idx, R2, T2_idx, K2_idx, R3, T3_idx, K3_idx, R4, T4_idx, K4_idx  = particle

            # Map indices to actual discrete values
            T1 = map_index_to_value(T1_idx, allowed_T1_values)
            T2 = map_index_to_value(T2_idx, allowed_T2_values)
            T3 = map_index_to_value(T3_idx, allowed_T3_values)
            T4 = map_index_to_value(T4_idx, allowed_T4_values)

            K1 = map_index_to_value(K1_idx, allowed_K1_values)
            K2 = map_index_to_value(K2_idx, allowed_K2_values)
            K3 = map_index_to_value(K3_idx, allowed_K3_values)
            K4 = map_index_to_value(K4_idx, allowed_K4_values)

            # Combine parameters into a structured format
            params = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3), (R4, T4, K4)]

            # Simulate RDII using the current parameters
            sim_rdii = RDII_calculation_4uh(params, delta_t, rainfall, area_acres)
            
            # # Compute RMSE
            # rmse = fitness(obs_rdii, sim_rdii, delta_t)
            sim_rdii = np.array(sim_rdii)

            # Check for invalid outputs
            if sim_rdii.size == 0 or np.any(np.isnan(sim_rdii)) or np.any(np.isinf(sim_rdii)):
                print("Invalid sim_rdii generated, returning large penalty.")
                return float('inf')

            # Pad the shorter array with zeros to match the length of the longer one
            max_length = max(len(obs_rdii), len(sim_rdii))
            obs_rdii_padded = np.pad(obs_rdii, (0, max_length - len(obs_rdii)), mode='constant')
            sim_rdii_padded = np.pad(sim_rdii, (0, max_length - len(sim_rdii)), mode='constant')

        
            # Calculate fitness value
            fitness_value = fitness(
                obs_rdii_padded,
                sim_rdii_padded,
                delta_t,
                weight_rmse=0.25,
                weight_r2=0.25,
                weight_pbias=0.25,
                weight_nse=0.25
            )
            

            # Apply penalties for constraints
            penalty = 0
            Ro = R_calc(rainfall, obs_rdii, delta_t, area_acres)
            if R1 + R2 + R3 >= Ro:
                penalty += 100 * (R1 + R2 + R3 - Ro)
            if not (T1 < T2 < T3):
                penalty += 100
            if not (T1 * K1 < T2 * K2 < T3 * K3):
                penalty += 100

            # Final fitness value (RMSE + penalties)
            total_fitness = fitness_value + penalty
            fitness_values.append(total_fitness)

        return np.array(fitness_values)  # Return fitness for all particles   

    # Define bounds for all parameters
    bounds = [
        (0, 1),  # R1
        (0, len(allowed_T1_values) - 1),  # T1 index
        (0, len(allowed_K1_values) - 1),  # K1 index
        (0, 1),  # R2
        (0, len(allowed_T2_values) - 1),  # T2 index
        (0, len(allowed_K2_values) - 1),  # K2 index
        (0, 1),  # R3
        (0, len(allowed_T3_values) - 1),  # T3 index
        (0, len(allowed_K3_values) - 1),  # K3 index
        (0, 1),  # R4
        (0, len(allowed_T4_values) - 1),  # T4 index
        (0, len(allowed_K4_values) - 1),  # K4 index
    ]  

    # Define the number of particles and iterations
    num_particles = 100
    num_iterations = 300

    # Initialize the PSO optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=num_particles,
        dimensions=len(bounds),
        options={"c1": 1.5, "c2": 1.5, "w": 0.7},
        bounds=(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])),  # Extract lower and upper bounds,
    )

    # Perform optimization
    best_cost, best_pos = optimizer.optimize(objective_function_pso_4uh, iters=num_iterations)

    # Map indices to actual T and K values
    R1, T1_index, K1_index, R2, T2_index, K2_index, R3, T3_index, K3_index, R4, T4_index, K4_index  = best_pos

    T1 = allowed_T1_values[int(T1_index)]
    T2 = allowed_T2_values[int(T2_index)]
    T3 = allowed_T3_values[int(T3_index)]
    T4 = allowed_T4_values[int(T4_index)]

    K1 = allowed_K1_values[int(K1_index)]
    K2 = allowed_K2_values[int(K2_index)]
    K3 = allowed_K3_values[int(K3_index)]
    K4 = allowed_K4_values[int(K4_index)]
    best_params_actual = [(R1, T1, K1), (R2, T2, K2), (R3, T3, K3) , (R4, T4, K4)]
    criteria= calculate_criteria(best_params_actual,obs_rdii,delta_t,rainfall,area_acres)
    best_fitness_value = best_cost

    criteria_df = pd.DataFrame({
        "Metric": ["RMSE", "R²", "Percent Bias (PBIAS)", "Nash-Sutcliffe Efficiency (NSE)"],
        "Value": [
            f"{criteria['RMSE']:.4f}",
            f"{criteria['R2']:.4f}",
            f"{criteria['PBIAS']:.2f}%",
            f"{criteria['NSE']:.4f}"
        ]
    }) 

    # Show results
    col1, col2  = st.columns([1,2])

    with col1:

        # DataFrame for best parameters
        best_params_df = pd.DataFrame(
            best_params_actual,
            columns=["R", "T [sec]", "K"],
            index=["Unit Hydrograph 1", "Unit Hydrograph 2", "Unit Hydrograph 3", "Unit Hydrograph 4"]
        ) 

        # Apply the conversion to the 'T [sec]' column
        best_params_df["T [hh:mm]"] = best_params_df["T [sec]"].apply(convert_seconds_to_hm)
        # Drop the original "T [sec]" column if not needed
        best_params_df = best_params_df.drop(columns=["T [sec]"])
        best_params_df = best_params_df[["R", "T [hh:mm]", "K"]]        

        # Display Best Parameters in a table
        st.write("**Optimized Parameters:**")
        st.table(best_params_df)

        # Display Criteria in a table
        st.write("**Metrices:**")
        st.table(criteria_df)

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                padding: 5px; 
                background-color: #f9f9f9; 
                color: #333; 
                text-align: center;
                font-size: 15px;">
                <b>Objective function Best Value:</b> {best_fitness_value}
            </div>
            """,
            unsafe_allow_html=True
        )         

    with col2:
        st.write("**Simulated and Observed RDII with Rainfall**")
        # Plot using RDII_calculation_and_plot
        RDII_calculation_and_plot_4uh(best_params_actual, delta_t, rainfall, area_acres, None, obs_rdii)
   
        st.write("**Unit Hydrographs**")
        plot_unit_hydrographs_with_sum(best_params_actual, delta_t) 




# Add a dropdown menu for selecting the optimization algorithm
st.subheader("Select Optimization Algorithm")
algorithm_options = [
    "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)",
    "Differential Evolution (DE)",
    "Genetic Algorithm (GA)",
    "Simulated Annealing (SA)",
    "Particle Swarm Optimization (PSO)"
]
selected_algorithm = st.selectbox("Choose an optimization algorithm:", algorithm_options)

# Display the selected algorithm
st.write(f"Selected Algorithm: {selected_algorithm}")

# #Execute based on the selected algorithm
# if st.button("Run Optimization"):
#     if selected_algorithm == "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)":
#         st.write("**Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**")
#         run_cma_es_3_uh()
#     elif selected_algorithm == "Differential Evolution (DE)":
#         st.write("**Differential Evolution (DE) optimization**")
#         # Add DE-specific logic
#         run_de_optimization()
#     elif selected_algorithm == "Genetic Algorithm (GA)":
#         st.write("**Genetic Algorithm (GA) optimization**")
#         run_ga_optimization()
#         # Add GA-specific logic
#     elif selected_algorithm == "Simulated Annealing (SA)":
#         st.write("**Simulated Annealing (SA)**")
#         run_sa_optimization()
#         # Add SA-specific logic
#     elif selected_algorithm == "Particle Swarm Optimization (PSO)":
#         st.write("**Particle Swarm Optimization (PSO) optimization**")
#         # Add PSO-specific logic
#         run_pso_optimization()

# Execute based on the selected algorithm
if st.button("Run Optimization"):
    if selected_algorithm == "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)":
        if num_unit_hydrographs == 2:
            st.write("**CMA-ES for 2 Unit Hydrographs**")
            run_cma_es_2_uh()
        elif num_unit_hydrographs == 3:
            st.write("**CMA-ES for 3 Unit Hydrographs**")
            run_cma_es_3_uh()
        elif num_unit_hydrographs == 4:
            st.write("**CMA-ES for 4 Unit Hydrographs**")
            run_cma_es_4_uh()

    elif selected_algorithm == "Differential Evolution (DE)":
        if num_unit_hydrographs == 2:
            st.write("**DE for 2 Unit Hydrographs**")
            run_de_2_uh()
        elif num_unit_hydrographs == 3:
            st.write("**DE for 3 Unit Hydrographs**")
            run_de_3_uh()
        elif num_unit_hydrographs == 4:
            st.write("**DE for 4 Unit Hydrographs**")
            run_de_4_uh()

    elif selected_algorithm == "Genetic Algorithm (GA)":
        if num_unit_hydrographs == 2:
            st.write("**GA for 2 Unit Hydrographs**")
            run_ga_2_uh()
        elif num_unit_hydrographs == 3:
            st.write("**GA for 3 Unit Hydrographs**")
            run_ga_3_uh()
        elif num_unit_hydrographs == 4:
            st.write("**GA for 4 Unit Hydrographs**")
            run_ga_4_uh()

    elif selected_algorithm == "Simulated Annealing (SA)":
        if num_unit_hydrographs == 2:
            st.write("**SA for 2 Unit Hydrographs**")
            run_sa_2_uh()
        elif num_unit_hydrographs == 3:
            st.write("**SA for 3 Unit Hydrographs**")
            run_sa_3_uh()
        elif num_unit_hydrographs == 4:
            st.write("**SA for 4 Unit Hydrographs**")
            run_sa_4_uh()

    elif selected_algorithm == "Particle Swarm Optimization (PSO)":
        if num_unit_hydrographs == 2:
            st.write("**PSO for 2 Unit Hydrographs**")
            run_pso_2_uh()
        elif num_unit_hydrographs == 3:
            st.write("**PSO for 3 Unit Hydrographs**")
            run_pso_3_uh()
        elif num_unit_hydrographs == 4:
            st.write("**PSO for 4 Unit Hydrographs**")
            run_pso_4_uh()