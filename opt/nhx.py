# nhx_dp.py

import numpy as np
import math
try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print("CoolProp이 설치되지 않았습니다. 'pip install CoolProp'로 설치하세요.")
    exit()

# -------------------- 헬퍼 함수 (기존과 동일) --------------------

def get_compressor_outlet(ref, p_in, T_in, p_out, eta_s):
    h_in = PropsSI('H', 'P', p_in, 'T', T_in, ref)
    s_in = PropsSI('S', 'P', p_in, 'T', T_in, ref)
    h_out_s = PropsSI('H', 'P', p_out, 'S', s_in, ref)
    
    if h_out_s < h_in:
        return h_in, h_in, T_in, False 

    h_out_actual = h_in + (h_out_s - h_in) / eta_s
    T_out_actual = PropsSI('T', 'P', p_out, 'H', h_out_actual, ref)
    return h_in, h_out_actual, T_out_actual, True

def get_cp(fluid, T_K, P_Pa=101325):
    try:
        return PropsSI('C', 'T', T_K, 'P', P_Pa, fluid)
    except ValueError:
        # 물성치 계산 실패 시 물의 비열과 유사한 값 반환
        if 'water' in fluid.lower():
            return 4180
        else:
            return 2000 

def calculate_required_UA(Q_actual, C_fluid, T_fluid_in, T_ref_sat):
    # 열교환기 UA 계산 (Effectiveness-NTU, C_r = 0 가정)
    delta_T_max = abs(T_fluid_in - T_ref_sat)
    
    if delta_T_max < 1e-3 or C_fluid < 1e-3: 
        return 1e9 # 분모가 0이 되는 것 방지
        
    Q_max = C_fluid * delta_T_max
    
    if Q_max < Q_actual:
        return 1e9 # 물리적으로 불가능 (필요 이상)
        
    epsilon_required = Q_actual / Q_max
    
    if epsilon_required >= 0.9999:
        return 1e9 # 무한대 UA 방지
    
    NTU_required = -math.log(1.0 - epsilon_required)
    UA_required = NTU_required * C_fluid
    return UA_required


# -------------------- (1단계) 사이클 평가 함수 (압력 강하 반영) --------------------

def evaluate_nhx_cycle(inputs, fixed_params):
    """
    Pymoo로부터 포화 압력(inputs)을 받아
    압력 강하(DP)를 적용한 사이클을 계산하고
    목표(COP, Total_UA)와 제약조건용 온도(T_evap, T_cond)를 반환
    """
    
    # 1. 입력 변수(포화 압력) 및 고정 변수 할당
    p_sat_evap_out = inputs[0] # 증발기 출구(압축기 입구) 포화 압력 (Pa)
    p_sat_cond_in = inputs[1]  # 응축기 입구(압축기 출구) 포화 압력 (Pa)
    
    ref = fixed_params["ref"]
    Q_H_target = fixed_params["Q_H_target"]
    T_source_in_K = fixed_params["T_source_in_K"]
    m_dot_source = fixed_params["m_dot_source"]
    T_sink_in_K = fixed_params["T_sink_in_K"]
    m_dot_sink = fixed_params["m_dot_sink"]
    superheat_K = fixed_params["superheat_K"]
    subcool_K = fixed_params["subcool_K"]
    eta_comp = fixed_params["eta_comp"]
    # (신규) 압력 강하 비율
    dp_ratio_gas = fixed_params["dp_ratio_gas"]
    dp_ratio_liquid = fixed_params["dp_ratio_liquid"]

    try:
        # 2. 제약 조건 계산용 기준 포화 온도
        # (p_sat_evap_out -> T_evap_K, p_sat_cond_in -> T_cond_K)
        T_evap_K = PropsSI('T', 'P', p_sat_evap_out, 'Q', 1, ref) 
        T_cond_K = PropsSI('T', 'P', p_sat_cond_in, 'Q', 1, ref)

        # 3. (신규) 압력 강하를 고려한 주요 상태점 압력 정의
        # p1: 압축기 입구 (증발기 출구)
        p1 = p_sat_evap_out
        # p2: 압축기 출구 (응축기 입구)
        p2 = p_sat_cond_in
        # p3: 팽창밸브 입구 (응축기 출구)
        p3 = p2 * (1 - dp_ratio_liquid)
        # p4: 팽창밸브 출구 (증발기 입구)
        # (p1 = p4 * (1 - dp_ratio_gas) -> p4 = p1 / (1 - dp_ratio_gas))
        p4 = p1 / (1 - dp_ratio_gas)
        
        # 4. 주요 상태점 물성치 계산
        
        # State 1: Compressor Inlet
        T1_K = T_evap_K + superheat_K
        h1 = PropsSI('H', 'T', T1_K, 'P', p1, ref)
        
        # State 2: Compressor Outlet
        h1_calc, h2, T2_K, comp_success = get_compressor_outlet(ref, p1, T1_K, p2, eta_comp)
        
        if not comp_success or h2 <= h1:
            return -1e-6, 1e9, 0.0, 0.0 # 페널티 반환

        # State 3: Expansion Valve Inlet
        # (p3에서의 포화 온도에서 subcool_K 만큼 과냉)
        T_sat_p3 = PropsSI('T', 'P', p3, 'Q', 0, ref)
        T3_K = T_sat_p3 - subcool_K
        h3 = PropsSI('H', 'T', T3_K, 'P', p3, ref)

        # State 4: Evaporator Inlet (p4, h4)
        h4 = h3 # 팽창밸브 (isoenthalpic)
        
        # 5. 냉매 유량 계산
        # (방열량은 압축기 출구(2) ~ 팽창밸브 입구(3) 사이에서 발생)
        Q_H_cycle_per_kg = h2 - h3
        if Q_H_cycle_per_kg <= 0:
            return -1e-6, 1e9, 0.0, 0.0
            
        m_dot_ref = Q_H_target / Q_H_cycle_per_kg

        # 6. [목표 1: COP] 계산
        W_comp = m_dot_ref * (h2 - h1)
        if W_comp <= 0:
            return -1e-9, 1e12, 0.0, 0.0
            
        COP = Q_H_target / W_comp
        
        # 7. [목표 2: Total UA] 계산
        
        # 증발기 (p4, h4 -> p1, h1)
        Q_evap_actual = m_dot_ref * (h1 - h4)
        Cp_source_water = get_cp('Water', T_source_in_K)
        C_source = m_dot_source * Cp_source_water
        # (UA 계산 시 기준 온도는 증발기 출구 포화 온도 T_evap_K 사용)
        UA_evap_req = calculate_required_UA(Q_evap_actual, C_source, T_source_in_K, T_evap_K)
        
        # 응축기 (p2, h2 -> p3, h3)
        Q_cond_actual = Q_H_target
        Cp_sink_water = get_cp('Water', T_sink_in_K)
        C_sink = m_dot_sink * Cp_sink_water
        # (UA 계산 시 기준 온도는 응축기 입구 포화 온도 T_cond_K 사용)
        UA_cond_req = calculate_required_UA(Q_cond_actual, C_sink, T_sink_in_K, T_cond_K)

        Total_UA = UA_evap_req + UA_cond_req
        
        # 8. 결과 반환 (Pymoo가 사용할 4개 값)
        # (목표 1, 목표 2, 제약조건 1, 제약조건 2)
        return -COP, Total_UA, T_evap_K, T_cond_K

    except ValueError as e:
        # CoolProp 오류 등 계산 실패 시 페널티 반환
        return -1e-6, 1e9, 0.0, 0.0
    

if __name__ == "__main__":
    
    # 1. (신규) 고정 변수 정의 (사용자 요청값 기준)
    fixed_params = {
        "ref": "R1233zdE",
        "Q_H_target": 200 * 1000,       # 200 kW
        "T_source_in_K": 75.0 + 273.15,
        "m_dot_source": 1.51,
        "T_sink_in_K": 100.0 + 273.15,
        "m_dot_sink": 3.0,
        "superheat_K": 5.0,
        "subcool_K": 2.0,
        "eta_comp": 0.75,
        "dp_ratio_gas": 0.05,
        "dp_ratio_liquid": 0.02
    }
    
    # 2. 테스트할 포화 온도 (섭씨)
    test_T_evap_C = 70.0 # (T_source 75.0 보다 낮아야 함)
    test_T_cond_C = 120.0 # (T_sink 100.0 보다 높아야 함)
    
    # 테스트 온도를 포화 압력(Pa)으로 변환
    test_p_evap = PropsSI('P', 'T', test_T_evap_C + 273.15, 'Q', 1, fixed_params["ref"])
    test_p_cond = PropsSI('P', 'T', test_T_cond_C + 273.15, 'Q', 1, fixed_params["ref"])

    test_inputs = [test_p_evap, test_p_cond]
    
    # 3. 함수 호출
    print(f"--- DP 적용 사이클 테스트 시작: T_evap={test_T_evap_C}°C, T_cond={test_T_cond_C}°C ---")
    print(f"    (입력 압력: P_evap_out={test_p_evap/1000:.1f} kPa, P_cond_in={test_p_cond/1000:.1f} kPa)")

    neg_COP, Total_UA, T_evap_K_out, T_cond_K_out = evaluate_nhx_cycle(test_inputs, fixed_params)
    
    # 4. 결과 출력
    actual_COP = -neg_COP
    
    print("\n--- 테스트 결과 ---")
    if actual_COP < 0.01:
        print("   > 계산 실패")
    else:
        print(f"  > (제약조건용) T_evap: {T_evap_K_out - 273.15:.3f} °C")
        print(f"  > (제약조건용) T_cond: {T_cond_K_out - 273.15:.3f} °C")
        print(f"  > 실제 COP: {actual_COP:.3f}")
        print(f"  > 필요 총 UA: {Total_UA:.2f} W/K")