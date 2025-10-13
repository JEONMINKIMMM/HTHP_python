"""
Single Stage Heat Pump Cycle with Inner Heat Exchanger

"""
from CoolProp.CoolProp import PropsSI
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Input Variables --------------------
ref = "R1233zdE"             # Refrigerant
T_source_in_C = 85.0         # Heat Source T-temperature (°C)
m_dot_source = 1.2           # Heat Source mass flow rate (kg/s)
T_sink_in_C = 100.0          # Heat Sink temperature (°C)
m_dot_sink = 1.0             # Heat Sink mass flow rate (kg/s)

# --- Target Values / Parameters ---
T_cond_target_C = 120.0      # ⭐ Condensor target temperature (°C)
T_evap_guess_C = 75.0        # Evaporator initial guess temperature (°C)
superheat_K = 5.0            # Superheat (K)
subcool_K = 2.0              # Subcool (K)
eta_comp = 0.75              # Compressor efficiency

# --- Pressure Losses ---
dp_ratio_gas = 0.05     # ⭐ Vapor pressure drop 5%
dp_ratio_liquid = 0.02  # ⭐ Liquid pressure drop 2%

# --- Heat Exchanger Specification ---
UA_evap = 3500.0             # Evaporator overall heat transfer coefficient * Area (W/K)
UA_cond = 4000.0             # Condensor overall heat transfer coefficient * Area (W/K)
UA_IHX = 250.0               # IHX overall heat transfer coefficient * Area (W/K)

# --- Convergence Set Up ---
max_iter = 200
tolerance = 1e-5
relaxation = 0.2

# -------------------- Helper Functions --------------------
def get_sat_properties(ref, T_K):
    p = PropsSI('P', 'T', T_K, 'Q', 0, ref)
    return p

def get_compressor_outlet(ref, p_in, T_in, p_out, eta_s):
    h_in = PropsSI('H', 'P', p_in, 'T', T_in, ref)
    s_in = PropsSI('S', 'P', p_in, 'T', T_in, ref)
    h_out_s = PropsSI('H', 'P', p_out, 'S', s_in, ref)
    h_out_actual = h_in + (h_out_s - h_in) / eta_s
    T_out_actual = PropsSI('T', 'P', p_out, 'H', h_out_actual, ref)
    return h_out_actual, T_out_actual

def get_cp(fluid, T_K, P_Pa=101325):
    return PropsSI('Cpmass', 'T', T_K, 'P', P_Pa, fluid)

def effectiveness_Cr_zero(NTU):
    return 1.0 - math.exp(-NTU)

def effectiveness_counterflow(NTU, Cr):
    if abs(Cr - 1.0) < 1e-6:
        return NTU / (1.0 + NTU)
    else:
        exp_val = math.exp(-NTU * (1.0 - Cr))
        return (1.0 - exp_val) / (1.0 - Cr * exp_val)

# -------------------- Initialization --------------------
T_source_in_K = T_source_in_C + 273.15
T_sink_in_K = T_sink_in_C + 273.15
T_cond_K = T_cond_target_C + 273.15
T_evap_K = T_evap_guess_C + 273.15
m_dot_ref = 0.5           # Guessed mass flow rate of refrigerant

print("--- Start the simulation ---")
print(f"Target of T_cond: {T_cond_target_C}°C, Initial guess of T_evap : {T_evap_guess_C}°C")

converged = False
for i in range(1, max_iter + 1):
    
    # 1. Saturation Pressure
    p_sat_evap = PropsSI('P', 'T', T_evap_K, 'Q', 1, ref)  # 증발기 포화압
    p_sat_cond = PropsSI('P', 'T', T_cond_K, 'Q', 1, ref)  # 응축기 포화압

    # 2. Pressure of State Points
    p1 = p_sat_evap                        # Evaporator outlet / IHX cold inlet
    p2 = p1 * (1 - dp_ratio_gas)           # IHX cold outlet / Compressor inlet
    p3 = p_sat_cond                        # Compressor outlet / Condenser inlet
    p4 = p3                                # Condenser outlet / IHX hot inlet
    p5 = p4 * (1 - dp_ratio_liquid)        # IHX hot outlet / EXV inlet
    p6 = p_sat_evap                        # EXV outlet / evaporator 

    # 3. IHX Inlet Enthalpy ( calculated by temperature )
    T1_K = T_evap_K + superheat_K
    h1 = PropsSI('H', 'T', T1_K, 'P', p1, ref)
    T4_K = T_cond_K - subcool_K
    h4 = PropsSI('H', 'T', T4_K, 'P', p4, ref)

    # 4. Evaporator
    Cp_source = m_dot_source * PropsSI('Cpmass', 'T', T_source_in_K, 'P', 101325, 'Water')
    NTU_evap = UA_evap / Cp_source
    epsilon_evap = effectiveness_Cr_zero(NTU_evap)
    Q_evap = epsilon_evap * Cp_source * (T_source_in_K - T_evap_K)  # [W]

    # 5. Safe Initial Estimation of m_dot_ref for the First Iteration
    if i == 1:
        # Calculate safe delta_h using sat.liquid enthalpy at the evaporator side
        h_liq_evap = PropsSI('H', 'T', T_evap_K, 'Q', 0, ref)
        delta_h_safe = h1 - h_liq_evap
        if delta_h_safe <= 1.0:      # If delta_h <= 1 J/kg, impossible -> set a lower bound
            delta_h_safe = max(delta_h_safe, 100.0)  # 100 J/kg lower bound
            print(f"Warning (iter1): small delta_h_safe used ({delta_h_safe:.1f} J/kg) to avoid div0")
        m_dot_ref = max(1e-4, Q_evap / delta_h_safe)
    else:
        # In subsequent iterations, using (h1 - h6) with IHX calculation 
        pass

    # 6. IHX Calculation (applied from the second iteration)
    Q_IHX = 0.0
    if i > 1:
        # IHX inlet states: cold vapor in = point1, hot liquid in = point4
        cp_vap = PropsSI('Cpmass', 'T', T1_K, 'P', p1, ref)
        cp_liq = PropsSI('Cpmass', 'T', T4_K, 'P', p4, ref)
        C_cold = m_dot_ref * cp_vap
        C_hot = m_dot_ref * cp_liq
        if C_cold <= 0 or C_hot <= 0:
            epsilon_IHX = 0.0
        else:
            C_min = min(C_cold, C_hot)
            C_max = max(C_cold, C_hot)
            Cr = C_min / C_max if C_max > 0 else 0.0
            NTU_IHX = UA_IHX / C_min if C_min > 0 else 0.0
            epsilon_IHX = effectiveness_counterflow(NTU_IHX, Cr)
            Q_IHX = epsilon_IHX * C_min * (T4_K - T1_K)  # positive means hot->cold
    else:
        epsilon_IHX = 0.0
        Q_IHX = 0.0

    # 7. IHX Outlet Enthalpy
    h2 = h1 + Q_IHX / m_dot_ref if m_dot_ref > 0 else h1
    h5 = h4 - Q_IHX / m_dot_ref if m_dot_ref > 0 else h4
    h6 = h5

    # 8. For Safety: h1 - h6 > 0
    delta_h_evap = h1 - h6
    if delta_h_evap <= 0:
        # If the denominator is <= 0, convergence X -> use a lower bound and print a warning,
        # and slightly increase T_evap to increase h1
        print(f"Warning (iter {i}): h1-h6 = {delta_h_evap:.2f} <= 0 -> Applying failsafe method")
        delta_h_evap = 100.0   # 100 J/kg lower bound
        # Also, nudge T_evap up by 0.5 K to change next iteration
        T_evap_K += 0.5

    # 9. Update Flow Rate of Refrigerant
    m_dot_ref = Q_evap / delta_h_evap

    # 10. Compressor
    try:
        T2_K = PropsSI('T', 'P', p2, 'H', h2, ref)
    except Exception as e:
        # If error in PropsSI, cannot calculate by p2 and h2 -> approximation by T1
        T2_K = T1_K
        print(f"Notice (iter {i}): PropsSI T2 failed -> use T1 as approximation. Err: {e}")

    h3, T3_K = get_compressor_outlet(ref, p2, T2_K, p3, eta_comp)
    W_compressor = m_dot_ref * (h3 - h2)

    # 11. Condensor
    Q_cond_required = Q_evap + W_compressor
    Cp_sink = m_dot_sink * PropsSI('Cpmass', 'T', T_sink_in_K, 'P', 101325, 'Water')
    NTU_cond = UA_cond / Cp_sink
    epsilon_cond = effectiveness_Cr_zero(NTU_cond)
    Q_cond_model = epsilon_cond * Cp_sink * (T_cond_K - T_sink_in_K)

    # 12. Residual Calculation & Update T_evap
    residual = Q_cond_model - Q_cond_required
    if i <= 10 or i % 10 == 0:
        print(f"Iter {i:3d}: T_evap={T_evap_K-273.15:6.2f}°C, Q_model={Q_cond_model/1e3:.2f}kW, Q_req={Q_cond_required/1e3:.2f}kW, Res={residual:8.2f}W, m_ref={m_dot_ref:.4f} kg/s")
    if abs(residual) < tolerance:
        print(f"\n>> Successfully Converged! (Iteration: {i})")
        converged = True
        break

    delta_T_evap_correction = residual / Cp_source
    T_evap_K -= relaxation * delta_T_evap_correction

    # If evaporation temp becomes higher than source temp, it's non-physical -> adjust
    if T_evap_K > T_source_in_K - 0.5:
        T_evap_K = T_source_in_K - 0.5

else:
    print(f"\n>> Reached the maximum number of iterations of ({max_iter}). Failed convergence.")


# -------------------- Results --------------------
def plot_thermodynamic_diagrams(df_results, ref_name):
    print("\n--- Creating Plots... ---")
    P_crit = PropsSI('Pcrit', ref_name)
    T_start = PropsSI('Ttriple', ref_name) + 2.0
    P_start = PropsSI('P', 'T', T_start, 'Q', 0, ref_name)
    p_dome = np.logspace(np.log10(P_start), np.log10(P_crit * 0.999), 200)

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
    plt.show()

if converged:
    print("\n--- Summary ---")
    print(f"Targted Condensation Saturation Temperature: {T_cond_K - 273.15:.2f} °C")
    print(f"Required Evaporation Saturation Temperature: {T_evap_K - 273.15:.2f} °C")
    print(f"Mass Flow Rate of Refrigerant: {m_dot_ref:.2f} kg/s")

    if W_compressor > 0:
        COP_H = Q_cond_required / W_compressor
        print(f"Heating COP: {COP_H:.3f}")
    else:
        print("Heating COP: N/A")

    # --- Properties of Each Points ---
    s1 = PropsSI('S', 'P', p1, 'H', h1, ref)
    s2 = PropsSI('S', 'P', p2, 'H', h2, ref)
    s3 = PropsSI('S', 'P', p3, 'H', h3, ref)
    s4 = PropsSI('S', 'P', p4, 'H', h4, ref)
    T5_K = PropsSI('T', 'P', p5, 'H', h5, ref)
    s5 = PropsSI('S', 'P', p5, 'H', h5, ref)
    T6_K = PropsSI('T', 'P', p6, 'H', h6, ref)
    s6 = PropsSI('S', 'P', p6, 'H', h6, ref)

    state_points_data = {
        'T [°C]': [T1_K-273.15, T2_K-273.15, T3_K-273.15, T4_K-273.15, T5_K-273.15, T6_K-273.15],
        'P [bar]': [p1/1e5, p2/1e5, p3/1e5, p4/1e5, p5/1e5, p6/1e5],
        'h [kJ/kg]': [h1/1e3, h2/1e3, h3/1e3, h4/1e3, h5/1e3, h6/1e3],
        's [kJ/kgK]': [s1/1e3, s2/1e3, s3/1e3, s4/1e3, s5/1e3, s6/1e3]
    }

    df_index = [
        '1. Evap Outlet', '2. Comp Inlet', '3. Comp Outlet',
        '4. Cond Outlet', '5. EXV Inlet', '6. EXV Outlet'
    ]
    df = pd.DataFrame(state_points_data, index=df_index)
    print("\n--- Physical Properties of State Points ---")
    print(df.round(3))

    plot_thermodynamic_diagrams(df, ref)
else:
    print("\n--- Failed to Converge the Simulation ---")
