import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve

# -------------------------
# Main cycle solver (No Flash, IHX only)
# -------------------------
def solve_IHX_cycle(
    ref="R1233zd(E)",
    Q_dot_heating=200e3,  # target heating capacity [W]
    T_cond_C=110.0,       # condenser temp (saturation, heat sink)
    T_evap_C=75.0,        # evaporator saturation temp (source)
    cond_SC=5.0,          # condenser subcooling [K]
    evap_SH=2.0,          # evaporator superheat [K]
    ihx_SH=5.0,           # vapor-side extra superheat (IHX)
    pinch_IHX=5.0,        # minimum approach temperature [K]
    eta_comp=0.75,        # compressor isentropic efficiency
    motor_eff=0.90,       # motor efficiency
    verbose=True
):

    # -------------------------
    # 1. Saturation states
    # -------------------------
    T_cond = T_cond_C + 273.15
    T_evap = T_evap_C + 273.15

    P_cond = PropsSI('P', 'T', T_cond, 'Q', 0, ref)
    P_evap  = PropsSI('P', 'T', T_evap, 'Q', 1, ref)

    # -------------------------
    # 2. Fixed points (condenser outlet & evaporator outlet)
    # -------------------------
    # State 4: condenser outlet (subcooled liquid)
    T4 = T_cond - cond_SC
    h4 = PropsSI('H', 'T', T4, 'P', P_cond, ref)
    s4 = PropsSI('S', 'T', T4, 'P', P_cond, ref)

    # State 1: evaporator outlet (slightly superheated)
    T1 = T_evap + evap_SH
    h1 = PropsSI('H', 'T', T1, 'P', P_evap, ref)
    s1 = PropsSI('S', 'T', T1, 'P', P_evap, ref)

    # -------------------------
    # 3. Define residual to match heating capacity
    # -------------------------
    def residual(m_ref):
        m = m_ref[0]

        # --- IHX balance ---
        # T2: after IHX superheating (compressor inlet)
        T2 = T1 + ihx_SH
        h2 = PropsSI('H', 'T', T2, 'P', P_evap, ref)

        # T4: liquid after IHX (further subcooled before expansion)
        # Energy balance in IHX: h5 - h4 = h2 - h1
        h5 = h4 - (h2 - h1)
        try:
            T5 = PropsSI('T', 'H', h5, 'P', P_cond, ref)
        except ValueError:
            return [1e6]  # unphysical

        # --- Expansion (4 → 6) ---
        h6 = h5  # throttling (isoenthalpic)
        P6 = P_evap
        T6 = PropsSI('T', 'H', h6, 'P', P_evap, ref)

        # --- Evaporator (6 → 1) ---
        Q_evap = m * (h1 - h6)

        # --- Compressor (2 → 3) ---
        s3 = PropsSI('S', 'T', T3, 'P', P_evap, ref)
        h3s = PropsSI('H', 'S', s3, 'P', P_cond, ref)
        h3 = h2 + (h3s - h2) / eta_comp
        W_comp = m * (h3 - h2)

        # --- Condenser (3 → 4) ---
        Q_cond = m * (h3 - h4)

        # Residual: match target heating capacity
        R = Q_cond - Q_dot_heating
        return [R]

    # -------------------------
    # 4. Solve for mass flow rate
    # -------------------------
    m_guess = Q_dot_heating / 1e5
    sol = fsolve(residual, [m_guess])
    m = sol[0]

    # -------------------------
    # 5. Recalculate with final m
    # -------------------------
    T2 = T1 + ihx_SH
    h2 = PropsSI('H', 'T', T2, 'P', P_evap, ref)
    h5 = h4 - (h2 - h1)
    T5 = PropsSI('T', 'H', h5, 'P', P_cond, ref)
    h6 = h5
    T6 = PropsSI('T', 'H', h6, 'P', P_evap, ref)

    s2 = PropsSI('S', 'T', T2, 'P', P_evap, ref)
    h3s = PropsSI('H', 'S', s2, 'P', P_cond, ref)
    h3 = h2 + (h3s - h2) / eta_comp
    T3 = PropsSI('T', 'H', h3, 'P', P_cond, ref)

    Q_cond = m * (h3 - h4)
    W_comp = m * (h3 - h2)
    COP_heating = Q_cond / (W_comp / motor_eff)

    results_dict = {
        'COP_Heating': COP_heating,
        'Q_Heating_kW': Q_cond / 1000,
        'm_kg_s': m,
        'IHX_dT_hot_cold': (T4 - T2) - pinch_IHX,
        'Comp_Inlet_T_C': T2 - 273.15,
        'Evap_T_C': T_evap_C,
    }

    return results_dict


# ===================================================
# Entry Point
# ===================================================
if __name__ == "__main__":
    T_evap_range_C = np.arange(50, 91, 5)
    results_list = []

    print("--- IHX 사이클 계산 시작 ---")
    for T_evap_loop in T_evap_range_C:
        print(f"T_evap_C = {T_evap_loop} °C 계산 중...")
        res = solve_IHX_cycle(T_evap_C=T_evap_loop, T_cond_C=120.0)
        res['T_evap_C'] = T_evap_loop
        results_list.append(res)

    df = pd.DataFrame(results_list)
    print("\n[결과 요약]")
    print(df)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(df['T_evap_C'], df['COP_Heating'], 'o-', label='COP (Heating)')
    plt.xlabel('Evaporator Saturation Temperature (°C)')
    plt.ylabel('Heating COP')
    plt.title('IHX Cycle Performance vs. Evaporator Temperature')
    plt.grid(True)
    plt.legend()

    ax2 = plt.twinx()
    ax2.plot(df['T_evap_C'], df['Q_Heating_kW'], 's--', color='red', label='Heating Capacity (kW)')
    ax2.set_ylabel('Heating Capacity (kW)')
    ax2.legend(loc='upper right')
    plt.show()
