import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve

# -------------------------
# Main cycle solver
# -------------------------
def solve_flash_ihx_cycle(
        ref='R1233zd(E)',
        Q_dot_heating=200e3,  # W
        T_cond_C=120.0,
        T_evap_C=75.0,
        P_int=None,
        cond_SC=5.0,
        evap_SH=2.0,
        ihx_SH=5.0,
        eta_comp1=0.75,
        eta_comp2=0.75,
        motor_eff=0.9,
        pinch=5.0,
        verbose=True
    ):

    ### ======= VALUE SETTINGS ======= ###
    # Convert temperatures to K
    T_cond = T_cond_C + 273.15 + pinch
    T_evap = T_evap_C + 273.15

    # Saturation pressures
    P_cond = PropsSI('P','T',T_cond,'Q',0,ref)
    P_evap = PropsSI('P','T',T_evap,'Q',1,ref)

    # Intermediate flash pressure
    if P_int is None:
        P_int = np.sqrt(P_evap * P_cond)

    ### ======= FIXED POINTS ======= ###
    # State 6: Condenser outlet (subcooled)
    T6 = T_cond - cond_SC
    P6 = P_cond
    h6 = PropsSI('H','T',T6,'P',P6,ref)

    # State 1 : Evaporator outlet (superheated)
    T1 = T_evap + evap_SH
    P1 = P_evap
    h1 = PropsSI('H','T',T1,'P',P1,ref)

    # State 11 : Flas tank - vapor outlet
    h11 = PropsSI('H', 'P', P_int, 'Q', 1, ref)
    T11 = PropsSI('T', 'H', h11, 'P', P_int, ref)

    # Solver function to find m_evap and flash vapor fraction y
    def residual(m_total_vars):
        m_total = m_total_vars[0] 
        
        if m_total <= 0:
            return 1e6 # 1개의 잔차만 반환
            
        # --- 1. 플래시 탱크 질량 분율 'y' 계산 ---
        # State 7 : EXV outlet (superheated)
        h7 = h6 # 등엔탈피 스로틀링
        
        # h8: 플래시 탱크 출구 (액체) - P_int의 포화액
        h8 = PropsSI('H', 'P', P_int, 'Q', 0, ref)
        
        # y: 플래시 증기 질량 분율 (m_vapor / m_total)
        # Quality x = (h - hf) / (hg - hf)
        y = (h7 - h8) / (h11 - h8)
        
        # y가 0~1 사이가 아니면 물리적으로 불가능
        if not (0 < y < 1):
            return 1e6

        m_flash_liquid = (1 - y) * m_total
        m_evap = m_flash_liquid
        m_flash_vapor = y * m_total

        # --- 2. IHX (내부 열교환기) 계산 ---
        # h1: 증발기 출구 (FIXED POINTS)
        # h2: IHX 출구 (Cold) / Comp 1 입구
        T2 = T1 + ihx_SH
        try:
            h2 = PropsSI('H', 'T', T2, 'P', P_evap, ref)
            s2 = PropsSI('S', 'T', T2, 'P', P_evap, ref)
        except ValueError:
            return 1e6

        # h10: 2차 팽창밸브 출구 / IHX 입구 (Hot)
        h10 = PropsSI('H', 'P', P_evap, 'Q', 0, ref)
        h8 = PropsSI('H', 'P', P_int, 'Q', 0, ref)
    
        # h9: IHX 출구 (Hot)
        # IHX 에너지 밸런스: m_evap * (h2 - h1) = m_evap * (h9 - h10)
        h9 = h8 - (h2 - h1)
        
        try:
            # h10이 유효한 액체 상태인지 확인
            PropsSI('T', 'H', h10, 'P', P_evap, ref) 
        except ValueError:
            return 1e6 # 액체 영역 벗어남

        # --- 3. 압축기 및 믹서 계산 ---
        
        # 압축기 1단 (h2 -> h3)
        h3s = PropsSI('H', 'S', s2, 'P', P_int, ref)
        h3 = h2 + (h3s - h2) / eta_comp1
        
        # 믹서 (h3 + h11 -> h4)
        # m_total * h4 = m_evap * h3 + m_flash_vapor * h11
        # m_total * h4 = (1-y)*m_total * h3 + y*m_total * h11
        h4 = (1 - y) * h3 + y * h11
        
        # 압축기 2단 (h4 -> h5)
        s4 = PropsSI('S', 'H', h4, 'P', P_int, ref)
        h5s = PropsSI('H', 'S', s4, 'P', P_cond, ref)
        h5 = h4 + (h5s - h4) / eta_comp2
        
        # --- 4. 응축기 및 잔차 계산 ---
        
        # 응축기 (Condenser)
        Q_cond_calc = m_total * (h5 - h6)
        
        # R1: 목표 난방 용량과의 차이
        R_Q = Q_cond_calc - Q_dot_heating
        
        return R_Q # 1개의 잔차만 반환

    # -------------------------------
    # Initial guess
    # -------------------------------
    m_guess = Q_dot_heating / 100e3 
    # [m_total] 1개의 추측값을 리스트로 전달
    vars_guess = [m_guess] 

    # -------------------------------
    # Solve
    # -------------------------------
    try:
        # 1변수 1잔차 방정식을 풂
        solution = fsolve(residual, vars_guess)
        m_total = solution[0] # 1개의 결과값을 받음
        
        # --- 수렴된 해(solution)를 바탕으로 최종 상태값들 재계산 ---
        
        # Flash Tank
        h7 = h6
        h8 = PropsSI('H', 'P', P_int, 'Q', 0, ref)
        # h11 from FIXED POINTS
        y = (h7 - h8) / (h11 - h8)
        
        if not (0 < y < 1):
             raise ValueError("Flash fraction 'y' is not between 0 and 1.")
             
        m_evap = (1 - y) * m_total
        m_flash_vapor = y * m_total

        # IHX
        T2 = T1 + ihx_SH 
        h2 = PropsSI('H', 'T', T2, 'P', P_evap, ref)
        s2 = PropsSI('S', 'T', T2, 'P', P_evap, ref)
        h9 = h8 - (h2 - h1)
        h10 = h9
        T10 = PropsSI('T', 'H', h10, 'P', P_evap, ref)
        T9 = PropsSI('T', 'H', h9, 'P', P_evap, ref)
        T8 = PropsSI('T', 'H', h8, 'P', P_int, ref)

        # Comps & Mixer
        h3s = PropsSI('H', 'S', s2, 'P', P_int, ref)
        h3 = h2 + (h3s - h2) / eta_comp1
        h4 = (1 - y) * h3 + y * h11
        s4 = PropsSI('S', 'H', h4, 'P', P_int, ref)
        h5s = PropsSI('H', 'S', s4, 'P', P_cond, ref)
        h5 = h4 + (h5s - h4) / eta_comp2
        
        # Performance
        Q_cond_calc = m_total * (h5 - h6)
        Q_evap_calc = m_evap * (h1 - h10)
        
        W_comp1 = m_evap * (h3 - h2)
        W_comp2 = m_total * (h5 - h4)
        W_total = (W_comp1 + W_comp2) / motor_eff
        
        COP_heating = Q_cond_calc / W_total
        
        if verbose:
            print("--- Cycle Solved (Flash Tank + IHX) ---")
            print(f"  Target Heating: {Q_dot_heating/1000:.2f} kW")
            print(f"  Solved Heating: {Q_cond_calc/1000:.2f} kW")
            print(f"  COP (Heating): {COP_heating:.3f}")
            print(f"  Total Mass Flow: {m_total:.3f} kg/s")
            print(f"  Evap Mass Flow: {m_evap:.3f} kg/s")
            print(f"  Flash Vapor Frac (y): {y:.3f}")
            print(f"  IHX SH (T2-T1): {T2-T1:.2f} K (Target: {ihx_SH} K)")
            print(f"  IHX Pinch (T9-T2): {T9-T2:.2f} K") # IHX 핀치 계산
            print(f"  Comp 1 Inlet Temp (T2): {T2-273.15:.2f} C")

        results_dict = {
            'COP_Heating': COP_heating,
            'Q_Heating_kW': Q_cond_calc / 1000,
            'm_total_kg_s': m_total,
            'y_fraction': y,
            'IHX_SH_K': T2-T1,
            'IHX_Pinch_K': T8-T2, # (T_hot_in - T_cold_out)
            'Comp1_Inlet_C': T2 - 273.15
        }
        return results_dict

    except Exception as e:
        if verbose:
            print(f"Error: Solver failed. {e}")
        return None
    
    

# ===================================================
# Example parameter sweep
# ===================================================
if __name__=='__main__':
    T_evap_range_C = np.arange(50,91,5)
    results_list = []

    for T_evap_loop in T_evap_range_C:
        res = solve_flash_ihx_cycle(T_cond_C=120.0, T_evap_C=T_evap_loop)
        results_list.append(res)

    results_df = pd.DataFrame(results_list)
    print(results_df)
