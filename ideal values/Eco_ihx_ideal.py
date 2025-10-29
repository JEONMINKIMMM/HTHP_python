import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve

# -------------------------
# Main cycle solver
# -------------------------
def solve_TS_FLASH_IHX_cycle(
    ref="R1233zd(E)",
    Q_dot_heating=200e3,  # target heating capacity [W]
    T_cond_C=110.0,       # condenser temp (saturation) in °C (heat sink)
    T_evap_C=75.0,        # evaporator saturation temp in °C (source side)
    P_int=None,           # intermediate (economizer/flash) pressure in Pa
    cond_SC=5.0,          # condenser subcooling
    evap_SH=2.0,          # evaporator superheat [K] (at evap outlet, before IHX)
    eco_SC=5.0,
    ihx_SH=5.0,
    eta_comp1=0.75,
    eta_comp2=0.75,
    motor_eff=0.90,
    pinch=5.0,            # IHX minimum approach (T_hot_in - T_cold_out)
    y = 0.3,
    verbose=True
):

    ### ======= VALUE SETTINGS ======= ###
    T_cond = T_cond_C + 273.15 + pinch
    T_evap = T_evap_C + 273.15

    # Saturation pressures
    P_high = PropsSI('P','T',T_cond,'Q',0,ref)
    P_low  = PropsSI('P','T',T_evap,'Q',1,ref)

    # Intermediate pressure (geometric mean if not given)
    if P_int is None:
        P_int = np.sqrt(P_high * P_low)

    ### ======= FIXED POINTS ======= ###
    # State 6: condenser outlet (subcooled liquid)
    T6 = T_cond - cond_SC
    P6 = P_high
    h6 = PropsSI('H','T',T6,'P',P6,ref)
    s6 = PropsSI('S','T',T6,'P',P6,ref)

    # State 1: evaporator outlet (saturated or slightly superheated)
    T1 = T_evap + evap_SH
    P1 = P_low
    h1 = PropsSI('H','T',T1,'P',P1,ref)
    s1 = PropsSI('S','T',T1,'P',P1,ref)

    # State 11: vapor outlet from economizer (flash gas)
    P11 = P_int
    h11 = PropsSI('H','P',P11,'Q',1,ref)
    s11 = PropsSI('S','P',P11,'Q',1,ref)
    T11 = PropsSI('T','P',P11,'Q',1,ref)

    ### ======= ECONOMIZER MASS FRACTION (y) ======= ###
    # State 7: economizer outlet (liquid to IHX) & IHX(hot) inlet
    T7 = T6 - eco_SC
    P7 = P_high
    h7 = PropsSI('H','T',T7,'P',P7,ref)
    s7 = PropsSI('S','T',T7,'P',P7,ref)

    # State 6m: EXV(sub) inlet
    h6m = h6
    P6m = P_high
    h10 = h6
    P10 = P_int

    def calc_eco(h10, h6, h7, m_evap, m_eco):
        return h10 + (m_evap / m_eco) * (h6 - h7)
        # y = m_eco / (m_evap + m_eco)
        # h11 = h10 + ((1-y)/y)*(h6 - h7)

### ======= SOLVER FUNCTION ======= ###
    # fsolve가 풀 함수를 정의합니다.
    # fsolve는 이 함수의 반환값이 [0, 0]이 되는 vars 값을 찾습니다.
    def residual(vars):
        m_evap, y = vars
        
        # --- 0. 유효성 검사 ---
        # m_evap (증발기 유량)은 0보다 커야 함
        # y (질량 분율)는 0과 1 사이여야 함
        if m_evap <= 0 or y <= 0 or y >= 1:
            return [1e6, 1e6]  # 유효하지 않은 추측값이면 큰 값을 반환
            
        # --- 1. 질량 유량 계산 ---
        # y = m_eco / m_total
        # m_evap = (1 - y) * m_total
        # m_total = m_evap / (1 - y)
        # m_eco = m_total * y = m_evap * y / (1 - y)
        m_eco = m_evap * y / (1 - y)
        m_total = m_evap + m_eco

        # --- 2. 이코노마이저(Subcooler) 계산 ---
        # 이 부분은 두 개의 잔차(residual) 중 하나를 구성합니다.
        
        # 보조 라인(m_eco) 상태
        # h9: 보조라인 팽창밸브 입구 (상태 6과 동일)
        h10 = h6 
        # h10: 보조라인 팽창밸브 출구 (P_int로 스로틀링)
        
        # h11: 보조라인 출구 (이코노마이저 통과 후)
        # -> 포화 증기(Q=1)가 된다고 가정 (이코노마이저 성능이 좋다고 가정)
        # (이것이 첫 번째 가정이자, 'y'를 결정하는 핵심)
        h11 = PropsSI('H', 'P', P_int, 'Q', 1, ref)
        T11 = PropsSI('T', 'H', h11, 'P', P_int, ref)

        # 주 라인(m_evap) 상태
        # h7: 주 라인 출구 (이코노마이저 통과 후)
        # 에너지 밸런스: m_evap * (h6 - h7) = m_eco * (h11 - h10)
        delta_h_eco = (m_eco / m_evap) * (h11 - h10)
        h7 = h6 - delta_h_eco
        
        # h7의 유효성 검사 (물리적으로 가능한지)
        try:
            T7 = PropsSI('T', 'H', h7, 'P', P_high, ref)
        except ValueError:
            return [1e6, 1e6] # 액체 영역을 벗어나면 에러

        # --- 3. 사이클 나머지 계산 (압축기, 응축기 등) ---
        T2 = T1 + ihx_SH 
        try:
            h2 = PropsSI('H', 'T', T2, 'P', P_low, ref)
            s2 = PropsSI('S', 'T', T2, 'P', P_low, ref)
        except ValueError:
             # T2가 T1보다 낮거나 유효하지 않은 상태일 경우
             return [1e6, 1e6]
        # h8: IHX Hot Out / Main EXV In (상태점 8)
        # IHX 에너지 밸런스: m_evap * (h7 - h8) = m_evap * (h2 - h1)
        # (m_evap는 양변에서 소거됨)
        h8 = h7 - (h2 - h1)
        
        try:
            T8 = PropsSI('T', 'H', h8, 'P', P_high, ref)
        except ValueError:
            return [1e6, 1e6] # h8이 액체 영역을 벗어남

        # --- 4. 사이클 나머지 계산 ---
        
        # h9: 주 라인 팽창밸브 출구 / 증발기 입구 (상태점 9)
        h9 = h8 # 스로틀링
        
        # (참고) 증발기 용량
        # Q_evap = m_evap * (h1 - h9) 
        
        # 압축기 1단 (h2 -> h_comp1_out)
        s3s = s2 # s1이 아니라 s2 (IHX 통과 후)
        h3s = PropsSI('H', 'S', s3s, 'P', P_int, ref)
        h3 = h2 + (h3s - h2) / eta_comp1
        
        # 믹서 (Mixer)
        # m_total * h4 = m_evap * h3 + m_eco * h11
        h4 = (m_evap * h3 + m_eco * h11) / m_total
        
        # 압축기 2단 (h4 -> h5)
        s4 = PropsSI('S', 'H', h4, 'P', P_int, ref)
        s5s = s4
        h5s = PropsSI('H', 'S', s5s, 'P', P_high, ref)
        h5 = h4 + (h5s - h4) / eta_comp2
        
        # 응축기 (Condenser)
        Q_cond_calc = m_total * (h5 - h6)
        
        # --- 5. 잔차(Residuals) 반환 ---
        
        # R1: 목표 난방 용량과의 차이
        R_Q = Q_cond_calc - Q_dot_heating
        
        # R2: 이코노마이저 핀치포인트(Pinch) 제약
        # (T_hot_out - T_cold_out) >= pinch
        R_pinch_eco = (T7 - T11) - pinch
        
        # (추가) IHX 핀치포인트 제약
        # IHX는 ihx_SH로 정의했으므로 핀치 제약은 일단 보류
        # T_hot_out(T8) > T_cold_in(T1) 인지만 확인
        if T8 <= T1:
            return [1e6, 1e6] # IHX 온도 역전

        return [R_Q, R_pinch_eco]

    # -------------------------------
    # Initial guess
    # -------------------------------
    m_guess = Q_dot_heating / 100e3 
    y_guess = 0.2 
    vars_guess = [m_guess, y_guess]

    # -------------------------------
    # Solve
    # -------------------------------
    try:
        solution = fsolve(residual, vars_guess)
        m_evap, y = solution
        
        # --- 수렴된 해(solution)를 바탕으로 최종 상태값들 재계산 ---
        m_eco = m_evap * y / (1 - y)
        m_total = m_evap + m_eco

        # Eco
        h11 = PropsSI('H', 'P', P_int, 'Q', 1, ref)
        T11 = PropsSI('T', 'H', h11, 'P', P_int, ref)
        delta_h_eco = (m_eco / m_evap) * (h11 - h10)
        h7 = h6 - delta_h_eco
        try:
            T7 = PropsSI('T', 'H', h7, 'P', P_high, ref)
        except ValueError:
            return [1e6, 1e6]
        
        # IHX
        T2 = T1 + ihx_SH 
        try:
            h2 = PropsSI('H', 'T', T2, 'P', P_low, ref)
        except ValueError:
            return [1e6, 1e6]
        s2 = PropsSI('S', 'T', T2, 'P', P_low, ref)
        h8 = h7 - (h2 - h1)
        try:
            T8 = PropsSI('T', 'H', h8, 'P', P_high, ref)
        except ValueError:
            return [1e6, 1e6]
        
        # Main line
        h9 = h8
        
        # Comps & Mixer
        s3s = s2
        h3s = PropsSI('H', 'S', s3s, 'P', P_int, ref)
        h3 = h2 + (h3s - h2) / eta_comp1
        
        h4 = (m_evap * h3 + m_eco * h11) / m_total
        
        s4 = PropsSI('S', 'H', h4, 'P', P_int, ref)
        s5s = s4
        h5s = PropsSI('H', 'S', s5s, 'P', P_high, ref)
        h5 = h4 + (h5s - h3) / eta_comp2
        
        # Performance
        Q_cond_calc = m_total * (h5 - h6)
        Q_evap_calc = m_evap * (h1 - h9)
        
        W_comp1 = m_evap * (h3 - h2)
        W_comp2 = m_total * (h4 - h3)
        W_total = (W_comp1 + W_comp2) / motor_eff
        
        COP_heating = Q_cond_calc / W_total
        
        
        results_dict = {
            'COP_Heating': COP_heating,
            'Q_Heating_kW': Q_cond_calc / 1000,
            'm_total_kg_s': m_total,
            'y_fraction': y,
            'Eco_Pinch_K': T7 - T11,
            'IHX_SH_K': T2 - T1,
            'IHX_Subcool_K': T7 - T8,
            'Comp1_Inlet_C': T2 - 273.15
        }
        return results_dict
        return COP_heating, Q_cond_calc

    except Exception as e:
        print(f"Error: Solver failed. {e}")
        return None
    
# ===================================================
# 스크립트 실행 지점 (Entry Point)
# ===================================================
if __name__ == "__main__":
    
    # 1. 시뮬레이션할 온도 범위 설정
    T_evap_range_C = np.arange(50, 91, 5)
    
    # 2. 결과를 저장할 리스트 생성
    results_list = []
    
    print("--- 파라미터 스터디 시작 (T_evap_C 변경) ---")
    
    # 3. for 루프를 사용하여 각 온도에 대해 함수 호출
    for T_evap_loop in T_evap_range_C:
        print(f"계산 중: T_evap_C = {T_evap_loop}°C ...")
        
        
        results = solve_TS_FLASH_IHX_cycle(
            T_cond_C = 120.0,
            T_evap_C = T_evap_loop,
            pinch = 3.0,      # 👈 (해결책) 핀치를 5K(기본값) 대신 2K로 완화
            verbose=False     # 루프 중에는 상세 정보 끄기
        )
        
        # 4. [수정] results가 None이 아닌지 (성공했는지) 확인
        if results is not None:
            # 성공한 경우, 기본 T_evap_C 값을 딕셔너리에 추가
            results['T_evap_C'] = T_evap_loop
            # 딕셔너리를 리스트에 추가
            results_list.append(results)
        else:
            # 솔버가 실패한 경우 (np.nan으로 빈 값 채우기)
            results_list.append({
                'T_evap_C': T_evap_loop,
                'COP_Heating': np.nan,
                'Q_Heating_kW': np.nan,
                'm_total_kg_s': np.nan,
                'y_fraction': np.nan,
                'Eco_Pinch_K': np.nan,
                'IHX_SH_K': np.nan,
                'IHX_Subcool_K': np.nan,
                'Comp1_Inlet_C': np.nan
            })

    print("--- 스터디 완료 ---")

    # 5. [수정] DataFrame이 이제 모든 열을 자동으로 표시
    results_df = pd.DataFrame(results_list)
    print("\n[시뮬레이션 결과]")
    print(results_df)
    
    # 6. (선택 사항) Matplotlib로 그래프 그리기 (코드는 동일함)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['T_evap_C'], results_df['COP_Heating'], 'o-', label='COP (Heating)')
    plt.xlabel('Evaporator Saturation Temperature (°C)')
    plt.ylabel('Heating COP')
    plt.title('Performance vs. Evaporator Temperature')
    plt.grid(True)
    plt.legend()
    
    ax2 = plt.twinx()
    ax2.plot(results_df['T_evap_C'], results_df['Q_Heating_kW'], 's--', color='red', label='Heating Capacity (kW)')
    ax2.set_ylabel('Heating Capacity (kW)')
    ax2.legend(loc='upper right')
    
    plt.show()