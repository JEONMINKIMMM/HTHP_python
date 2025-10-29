"""
Single Stage Heat Pump Cycle without Heat Exchanger

"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# -------------------- Input Variables --------------------
ref = "R1233zdE"              # Refrigerant
T_source_in_C = 75.0          # Heat Source T-temperature (°C)
m_dot_source = 1.51           # Heat Source mass flow rate (kg/s)
T_sink_in_C = 100.0           # Heat Sink temperature (°C)
m_dot_sink = 1.0              # Heat Sink mass flow rate (kg/s)

# --- Target Values / Parameters ---
T_cond_target_C = 120.0       # ⭐ Condensor target temperature (°C)
T_evap_guess_C = 75.0         # Evaporator initial guess temperature (°C)
superheat_K = 5.0             # Superheat (K)
subcool_K = 2.0               # Subcool (K)
eta_comp = 0.75               # Compressor efficiency

# --- Pressure Losses ---
dp_ratio_gas = 0.05           # ⭐ Vapor pressure drop 5%
dp_ratio_liquid = 0.02        # ⭐ Liquid pressure drop 2%

# --- Heat Exchanger Specification ---
UA_evap = 3500.0              # Evaporator overall heat transfer coefficient * Area (W/K)
UA_cond = 4000.0              # Condensor overall heat transfer coefficient * Area (W/K)

# --- Convergence Set Up ---
max_iter = 200
tolerance = 1e-5
relaxation = 0.2

# -------------------- Helper Functions --------------------

def get_compressor_outlet(ref, p_in, T_in, p_out, eta_s):
    h_in = PropsSI('H', 'P', p_in, 'T', T_in, ref)
    s_in = PropsSI('S', 'P', p_in, 'T', T_in, ref)
    h_out_s = PropsSI('H', 'P', p_out, 'S', s_in, ref)
    h_out_actual = h_in + (h_out_s - h_in) / eta_s
    T_out_actual = PropsSI('T', 'P', p_out, 'H', h_out_actual, ref)
    return h_in, h_out_actual, T_out_actual

def get_cp(fluid, T_K, P_Pa=101325):
    return PropsSI('C', 'T', T_K, 'P', P_Pa, fluid)

def effectiveness_Cr_zero(NTU):
    return 1.0 - math.exp(-NTU)

# -------------------- Initialization --------------------
T_source_in_K = T_source_in_C + 273.15
T_sink_in_K = T_sink_in_C + 273.15
T_cond_K = T_cond_target_C + 273.15
T_evap_K = T_evap_guess_C + 273.15

print("--- Start the simulation ---")
print(f"Target of T_cond: {T_cond_target_C}°C, Initial guess of T_evap : {T_evap_guess_C}°C")

# -------------------- Main Loop --------------------
converged = False
for i in range(1, max_iter + 1):
    
    # 1. Saturation Pressure
    p_sat_evap = PropsSI('P', 'T', T_evap_K, 'Q', 1, ref)  # 증발기 포화압
    p_sat_cond = PropsSI('P', 'T', T_cond_K, 'Q', 1, ref)  # 응축기 포화압

    # 2. Pressure of State Points
    p1 = p_sat_evap                        # Evaporator outlet / Compressor inlet
    p2 = p_sat_cond                        # Compressor outlet / Condenser inlet
    p3 = p2 * (1 - dp_ratio_liquid)        # Condenser outlet / EXV inlet
    p4 = p1 / (1 - dp_ratio_gas)    # EXV outlet / Evaporator inlet

    # 2. Define State Points
    T1_K = T_evap_K + superheat_K
    h1 = PropsSI('H', 'T', T1_K, 'P', p1, ref)
    T3_K = T_cond_K - subcool_K
    h3 = PropsSI('H', 'T', T3_K, 'P', p3, ref)
    h4 = h3

    # 3. Evaporator
    Cp_source = m_dot_source * get_cp('Water', T_source_in_K)
    C_min_evap = Cp_source
    NTU_evap = UA_evap / C_min_evap
    epsilon_evap = effectiveness_Cr_zero(NTU_evap)
    Q_evap = epsilon_evap * C_min_evap * (T_source_in_K - T_evap_K)

    # 4. Mass Flow Rate of Refrigerant
    if (h1 - h4) <= 0:
        print("Error: h1-h4 <= 0.")
        converged = False
        break
    m_dot_ref = Q_evap / (h1 - h4)

    # 5. Compressor Performance
    h1_calc, h2, T2_K = get_compressor_outlet(ref, p1, T1_K, p2, eta_comp)
    W_compressor = m_dot_ref * (h2 - h1)

    # 6. Condensor
    Q_cond_required = Q_evap + W_compressor
    Cp_sink = m_dot_sink * get_cp('Water', T_sink_in_K)
    C_min_cond = Cp_sink
    NTU_cond = UA_cond / C_min_cond
    epsilon_cond = effectiveness_Cr_zero(NTU_cond)
    Q_cond_model = epsilon_cond * C_min_cond * (T_cond_K - T_sink_in_K)

    # 7. Residual Calculation & Update T_evap
    residual = Q_cond_model - Q_cond_required
    if i <= 10 or i % 10 == 0:
        print(f"Iter {i:3d}: T_evap={T_evap_K-273.15:6.2f}°C, Q_model={Q_cond_model/1e3:6.2f}kW, Q_req={Q_cond_required/1e3:6.2f}kW, Res={residual:8.2f}W")
    if abs(residual) < tolerance:
        print(f"\n>> Successfully Converged! (Iteration: {i})")
        converged = True
        break
    delta_T_evap_correction = residual / C_min_evap
    T_evap_K -= relaxation * delta_T_evap_correction
else:
    print(f"\n>> Reached the maximum number of iterations of ({max_iter}). Failed convergence.")

# -------------------- Results --------------------
def plot_thermodynamic_diagrams(df_results, ref_name, save_path=None):
    print("\n--- Creating Plots... ---")
    P_crit = PropsSI('Pcrit', ref_name)
    T_start = T_evap_K - 30
    P_start = PropsSI('P', 'T', T_start, 'Q', 0, ref_name)
    p_dome = np.logspace(np.log10(P_start), np.log10(P_crit * 0.999), 200)

    # Calculate Dew Point
    p_cond_sat = PropsSI('P', 'T', T_cond_K, 'Q', 1, ref_name)
    T_dew_point = PropsSI('T', 'P', p_cond_sat, 'Q', 1, ref_name)
    h_dew_point = PropsSI('H', 'P', p_cond_sat, 'Q', 1, ref_name)
    s_dew_point = PropsSI('S', 'P', p_cond_sat, 'Q', 1, ref_name)
    
    T_liq, T_vap, h_liq, h_vap, s_liq, s_vap = [], [], [], [], [], []
    for p in p_dome:
        try:
            T_liq.append(PropsSI('T', 'P', p, 'Q', 0, ref_name))
            T_vap.append(PropsSI('T', 'P', p, 'Q', 1, ref_name))
            h_liq.append(PropsSI('H', 'P', p, 'Q', 0, ref_name))
            h_vap.append(PropsSI('H', 'P', p, 'Q', 1, ref_name))
            s_liq.append(PropsSI('S', 'P', p, 'Q', 0, ref_name))
            s_vap.append(PropsSI('S', 'P', p, 'Q', 1, ref_name))
        except ValueError:
            continue

    T_liq_C, T_vap_C = np.array(T_liq) - 273.15, np.array(T_vap) - 273.15
    p_dome_bar = np.array(p_dome) / 1e5
    h_liq_kJ, h_vap_kJ = np.array(h_liq) / 1e3, np.array(h_vap) / 1e3
    s_liq_kJ, s_vap_kJ = np.array(s_liq) / 1e3, np.array(s_vap) / 1e3

    h_cycle = df_results['h [kJ/kg]'].tolist(); h_cycle.append(h_cycle[0])
    p_cycle = df_results['P [bar]'].tolist(); p_cycle.append(p_cycle[0])
    s_cycle = df_results['s [kJ/kgK]'].tolist(); s_cycle.append(s_cycle[0])
    T_cycle = df_results['T [°C]'].tolist(); T_cycle.append(T_cycle[0])

    s_cycle.insert(2, s_dew_point / 1e3)
    T_cycle.insert(2, T_dew_point-273.15)
    p_cycle.insert(2, p_cond_sat / 1e5)
    h_cycle.insert(2,h_dew_point / 1e3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.plot(h_liq_kJ, p_dome_bar, 'b-', label='Saturated Liquid')
    ax1.plot(h_vap_kJ, p_dome_bar, 'r-', label='Saturated Vapor')
    ax1.plot(h_cycle, p_cycle, 'ko-', mfc='lime', label='HP Cycle')
    ax1.set_yscale('log'); ax1.set_title(f'P-h Diagram ({ref_name})')
    ax1.set_xlabel('Enthalpy [kJ/kg]'); ax1.set_ylabel('Pressure [bar]')
    ax1.grid(True, which="both", ls="--"); ax1.legend()
    for i, txt in enumerate(df_results.index):
        ax1.annotate(txt.split('.')[0], (df_results['h [kJ/kg]'][i]+5, df_results['P [bar]'][i]*1.05))

    ax2.plot(s_liq_kJ, T_liq_C, 'b-', label='Saturated Liquid')
    ax2.plot(s_vap_kJ, T_vap_C, 'r-', label='Saturated Vapor')
    ax2.plot(s_cycle, T_cycle, 'ko-', mfc='lime', label='HP Cycle')
    ax2.set_title(f'T-s Diagram ({ref_name})'); ax2.set_xlabel('Entropy [kJ/kgK]')
    ax2.set_ylabel('Temperature [°C]'); ax2.grid(True, ls="--"); ax2.legend()
    for i, txt in enumerate(df_results.index):
        ax2.annotate(txt.split('.')[0], (df_results['s [kJ/kgK]'][i]+0.01, df_results['T [°C]'][i]+3))
    
    plt.tight_layout()
    
    # Save the figure BEFORE showing it
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        
    plt.show()

if converged:
    print("\n--- Summary ---")
    print(f"Targeted Condensation Saturation Temperature: {T_cond_K - 273.15:.2f} °C")
    print(f"Required Evaporation Saturation Temperature: {T_evap_K - 273.15:.2f} °C")
    print(f"Mass Flow Rate of Refrigerant: {m_dot_ref:.3f} kg/s")
    
    if W_compressor > 0:
        COP_H = Q_cond_required / W_compressor
        density = PropsSI('D','T', T2_K ,'P', p2, ref)
        v_in = 1 / density
        VHC = Q_cond_required / (m_dot_ref * v_in)        #J/m^3
        
        print(f"Heating COP: {COP_H:.3f} (-)")
        print(f"Volumetric Heating Capacity (VHC): {VHC/1e3:.3f} kJ/m^3")
        print(f"Compressor Power Consumption: {W_compressor/1e3:.3f} kW")
    else:
        print("Heating COP: N/A")

    # --- Properties of Each Points ---
    s1 = PropsSI('S', 'P', p1, 'H', h1, ref)
    s2 = PropsSI('S', 'P', p2, 'H', h2, ref)
    s3 = PropsSI('S', 'P', p3, 'H', h3, ref)
    T4_K = PropsSI('T', 'P', p4, 'H', h4, ref)
    s4 = PropsSI('S', 'P', p4, 'H', h4, ref)

    # --- Create pandas DataFrame ---
    state_points_data = {
        'T [°C]': [T1_K - 273.15, T2_K - 273.15, T3_K - 273.15, T4_K - 273.15],
        'P [bar]': [p1 / 1e5, p2 / 1e5, p3 / 1e5, p4 / 1e5],
        'h [kJ/kg]': [h1 / 1e3, h2 / 1e3, h3 / 1e3, h4 / 1e3],
        's [kJ/kgK]': [s1 / 1e3, s2 / 1e3, s3 / 1e3, s4 / 1e3]
    }
    df = pd.DataFrame(state_points_data, index=['1. Comp Inlet', '2. Comp Outlet', '3. EXV Inlet', '4. EXV Outlet'])

    print("\n--- Physical Properties of State Points ---")
    print(df.round(3))
    
    # -------------------- Save Results --------------------
    print("\n--- Saving Results ---")
    output_dir = "results"  # Save results in a subfolder named 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # Create the folder if it doesn't exist

    # Define file paths using os.path.join for compatibility
    summary_path = os.path.join(output_dir, "nhx_summary.txt")
    csv_path = os.path.join(output_dir, "nhx_cycle_result.csv")
    plot_path = os.path.join(output_dir, "nhx_performance_plot.png")

    # (1) Save text summary
    with open(summary_path, "w") as f:
        f.write(f"Targeted Condensation Saturation Temperature: {T_cond_K - 273.15:.2f} °C\n")
        f.write(f"Required Evaporation Saturation Temperature: {T_evap_K - 273.15:.2f} °C\n")
        f.write(f"Mass Flow Rate of Refrigerant: {m_dot_ref:.3f} kg/s\n")
        f.write(f"Heating COP: {COP_H:.3f}\n\n")
        f.write(f"Volumetric Heating Capacity (VHC): {VHC/1e3:.3f} kJ/m^3\n\n")
        f.write(f"Compressor Power Consumption: {W_compressor/1e3:.3f} kW\n\n")
        f.write("--- Physical Properties ---\n")
        f.write(df.round(3).to_string())
    print(f"Summary saved to {summary_path}")

    # (2) Save CSV results
    df.round(3).to_csv(csv_path)
    print(f"CSV data saved to {csv_path}")

    # (3) Call the plotting function to generate and save the diagrams
    plot_thermodynamic_diagrams(df, ref, save_path=plot_path)

else:
    print("\n--- Failed to Converge the Simulation ---")
    print("--- No results were saved. ---")
