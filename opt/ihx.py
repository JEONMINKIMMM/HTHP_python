import numpy as np
import math
from scipy.optimize import fsolve

try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print("CoolProp이 설치되지 않았습니다. 'pip install CoolProp'로 설치하세요.")
    exit()

# -------------------- 헬퍼 함수 --------------------

def get_compressor_outlet(ref, p_in, T_in, p_out, eta_s):
    """압축기 출구 상태 계산"""
    h_in = PropsSI('H', 'P', p_in, 'T', T_in, ref)
    s_in = PropsSI('S', 'P', p_in, 'T', T_in, ref)
    h_out_s = PropsSI('H', 'P', p_out, 'S', s_in, ref)
    
    if h_out_s < h_in:
        return h_in, h_in, T_in, False 

    h_out_actual = h_in + (h_out_s - h_in) / eta_s
    T_out_actual = PropsSI('T', 'P', p_out, 'H', h_out_actual, ref)
    return h_in, h_out_actual, T_out_actual, True

def get_cp(fluid, T_K, P_Pa=101325):
    """비열(Cp) 계산"""
    try:
        return PropsSI('Cpmass', 'T', T_K, 'P', P_Pa, fluid)
    except ValueError:
        if 'water' in fluid.lower():
            return 4180
        else:
            return 2000 

def get_NTU_from_epsilon(epsilon, Cr, hx_type="counterflow"):
    """
    유용도(epsilon)와 용량비(Cr)로부터 NTU를 역계산
    (Cr=0인 경우는 calculate_required_UA에서 처리)
    """
    if epsilon >= 0.9999:
        return 1e9 # 무한대 NTU
    if epsilon <= 1e-9:
        return 0.0 # NTU 0
        
    try:
        if hx_type == "counterflow":
            if abs(Cr - 1.0) < 1e-6:
                # Epsilon = NTU / (1 + NTU) -> NTU = Epsilon / (1 - Epsilon)
                return epsilon / (1.0 - epsilon)
            else:
                # Epsilon = [1 - exp(-NTU(1-Cr))] / [1 - Cr * exp(-NTU(1-Cr))]
                num = math.log((epsilon - 1.0) / (epsilon * Cr - 1.0))
                den = Cr - 1.0
                return num / den
        else: # parallel
            den = 1.0 + Cr
            return -math.log(1.0 - epsilon * den) / den
            
    except (ValueError, OverflowError):
         return 1e12 # 페널티 (비물리적 Epsilon)


def calculate_required_UA(
    Q_actual, 
    C_hot, T_hot_in_K, 
    C_cold, T_cold_in_K, 
    phase_change=None
):
    """
    범용 UA 계산 함수 (상변화만, Cr=0 가정)
    """
    
    delta_T_max = abs(T_hot_in_K - T_cold_in_K)
    
    if delta_T_max < 1e-3:
        return 1e12 # 페널티 값

    C_min = 0.0
    
    if phase_change == "hot": # 응축기
        C_min = C_cold
        delta_T_max = T_hot_in_K - T_cold_in_K
    elif phase_change == "cold": # 증발기
        C_min = C_hot
        delta_T_max = T_hot_in_K - T_cold_in_K
    else:
        return 1e12 # 이 함수는 상변화 전용

    if C_min < 1e-6 or delta_T_max < 1e-3:
         return 1e12

    Q_max = C_min * delta_T_max

    if Q_max <= 0 or Q_max < (Q_actual - 1e-3):
        return 1e12 
        
    epsilon = Q_actual / Q_max
    
    if epsilon >= 0.9999:
        return 1e12

    try:
        # C_r = 0 (상변화)
        NTU_required = -math.log(1.0 - epsilon)
    except (ValueError, OverflowError):
         return 1e12

    UA_required = NTU_required * C_min
    
    if UA_required < 0 or not math.isfinite(UA_required):
         return 1e12

    return UA_required

# -------------------- (1단계) 사이클 평가 함수 (IHX Epsilon 입력) --------------------
def evaluate_ihx_cycle(inputs, fixed_params):
    """
    Pymoo로부터 [P_evap, P_cond, epsilon_ihx] (inputs)을 받아
    IHX 사이클을 계산하고
    목표(COP, Total_UA=Evap+Cond+IHX)와 제약조건용 온도를 반환
    """
    
    try:
        # 1. 입력 변수 및 고정 변수 할당
        p_sat_evap = inputs[0] # 증발기 출구(1) 포화 압력 (Pa)
        p_sat_cond = inputs[1] # 응축기 입구(3) 포화 압력 (Pa)
        epsilon_ihx = inputs[2] # IHX 유용도 (0.0 ~ 0.99)
        
        ref = fixed_params["ref"]
        Q_H_target = fixed_params["Q_H_target"]
        T_source_in_K = fixed_params["T_source_in_K"]
        m_dot_source = fixed_params["m_dot_source"]
        T_sink_in_K = fixed_params["T_sink_in_K"]
        m_dot_sink = fixed_params["m_dot_sink"]
        superheat_K = fixed_params["superheat_K"]
        subcool_K = fixed_params["subcool_K"]
        eta_comp = fixed_params["eta_comp"]
        dp_ratio_gas = fixed_params["dp_ratio_gas"]
        dp_ratio_liquid = fixed_params["dp_ratio_liquid"]
    
        # 2. 제약 조건 계산용 기준 포화 온도
        T_evap_K = PropsSI('T', 'P', p_sat_evap, 'Q', 1, ref) 
        T_cond_K = PropsSI('T', 'P', p_sat_cond, 'Q', 1, ref)

        # 3. 압력 강하를 고려한 6개 상태점 압력
        p1 = p_sat_evap
        p2 = p1 * (1 - dp_ratio_gas)
        p3 = p_sat_cond
        p4 = p3 * (1 - dp_ratio_liquid)
        p5 = p4 * (1 - dp_ratio_liquid)
        p6 = p1 / (1 - dp_ratio_gas)
        
        # 4. IHX 입구 상태 (T, P -> h)
        T1_K = T_evap_K + superheat_K
        h1 = PropsSI('H', 'T', T1_K, 'P', p1, ref) # IHX Cold In
        T4_K = T_cond_K - subcool_K
        h4 = PropsSI('H', 'T', T4_K, 'P', p4, ref) # IHX Hot In
        
        # IHX 작동 불가 조건 (입구 온도 역전)
        if T4_K <= T1_K and epsilon_ihx > 0.0:
            return -1e-9, 1e12, 0.0, 0.0 # 페널티

        # 5. 냉매 유량(m_dot_ref) 계산 (fsolve 사용)
        
        def residual_m_dot(m_dot_guess_array):
            m_dot = m_dot_guess_array[0]
            if m_dot <= 1e-6: return 1e6
                
            try:
                # 5a. IHX 열량(Q_IHX) 계산
                cp_vap = get_cp(ref, (T1_K + T4_K)/2.0, p1) # 근사
                cp_liq = get_cp(ref, (T1_K + T4_K)/2.0, p4) # 근사
                C_cold_ihx = m_dot * cp_vap
                C_hot_ihx = m_dot * cp_liq
                C_min_ihx = min(C_cold_ihx, C_hot_ihx)
                
                Q_IHX = epsilon_ihx * C_min_ihx * (T4_K - T1_K)
                if Q_IHX < 0: Q_IHX = 0.0

                # 5b. IHX 출구 엔탈피 (압축기 입구 h2, 팽창밸브 입구 h5)
                h2 = h1 + Q_IHX / m_dot # Comp Inlet
                h5 = h4 - Q_IHX / m_dot # EXV Inlet
                
                # 5c. 압축기 출구 엔탈피 (h3)
                T2_K = PropsSI('T', 'P', p2, 'H', h2, ref)
                _, h3, _, comp_success = get_compressor_outlet(ref, p2, T2_K, p3, eta_comp)
                
                if not comp_success or h3 <= h2: return 1e6

                # 5d. Q_H_target을 만족하는 필요 유량 계산
                Q_H_cycle_per_kg = h3 - h4 # 응축기 방열량 (IHX 이전)
                if Q_H_cycle_per_kg <= 1e-3: return 1e6
                
                m_dot_required = Q_H_target / Q_H_cycle_per_kg
                
                return m_dot - m_dot_required
                
            except Exception:
                return 1e6

        # 5e. m_dot_ref 풀이
        h1_simple, h2_simple, _, _ = get_compressor_outlet(ref, p1, T1_K, p3, eta_comp)
        m_guess = Q_H_target / (h2_simple - h4) if (h2_simple - h4) > 0 else 0.5
        
        sol = fsolve(residual_m_dot, [m_guess], xtol=1e-5)
        m_dot_ref = sol[0]

        if abs(residual_m_dot([m_dot_ref])) > 1e-3 or m_dot_ref <= 1e-6:
            raise ValueError("Failed to solve for refrigerant mass flow.")

        # 6. 최종 사이클 상태 재계산 (수렴된 m_dot_ref 사용)
        cp_vap = get_cp(ref, (T1_K + T4_K)/2.0, p1)
        cp_liq = get_cp(ref, (T1_K + T4_K)/2.0, p4)
        C_cold_ihx = m_dot_ref * cp_vap
        C_hot_ihx = m_dot_ref * cp_liq
        C_min_ihx = min(C_cold_ihx, C_hot_ihx)
        
        Q_IHX = epsilon_ihx * C_min_ihx * (T4_K - T1_K)
        if Q_IHX < 0: Q_IHX = 0.0

        h2 = h1 + Q_IHX / m_dot_ref
        h5 = h4 - Q_IHX / m_dot_ref
        h6 = h5
        
        T2_K = PropsSI('T', 'P', p2, 'H', h2, ref)
        _, h3, T3_K, _ = get_compressor_outlet(ref, p2, T2_K, p3, eta_comp)

        # 7. [목표 1: COP] 계산
        W_comp = m_dot_ref * (h3 - h2)
        if W_comp <= 0:
            return -1e-9, 1e12, 0.0, 0.0
        COP = Q_H_target / W_comp
        
        # 8. [목표 2: Total UA] 계산 (Evap + Cond + IHX)
        
        # 8a. UA_evap 계산
        Q_evap_actual = m_dot_ref * (h1 - h6)
        C_source = m_dot_source * get_cp('Water', T_source_in_K)
        UA_evap_req = calculate_required_UA(
            Q_actual=Q_evap_actual,
            C_hot=C_source, T_hot_in_K=T_source_in_K,
            C_cold=float('inf'), T_cold_in_K=T_evap_K,
            phase_change="cold"
        )
        
        # 8b. UA_cond 계산
        Q_cond_actual = Q_H_target 
        C_sink = m_dot_sink * get_cp('Water', T_sink_in_K)
        UA_cond_req = calculate_required_UA(
            Q_actual=Q_cond_actual,
            C_hot=float('inf'), T_hot_in_K=T_cond_K,
            C_cold=C_sink, T_cold_in_K=T_sink_in_K,
            phase_change="hot"
        )

        # 8c. UA_IHX 계산
        UA_IHX_req = 0.0
        if epsilon_ihx > 1e-6 and C_min_ihx > 1e-6:
            C_max_ihx = max(C_cold_ihx, C_hot_ihx)
            Cr_ihx = C_min_ihx / C_max_ihx
            NTU_ihx = get_NTU_from_epsilon(epsilon_ihx, Cr_ihx, "counterflow")
            UA_IHX_req = NTU_ihx * C_min_ihx

        Total_UA = UA_evap_req + UA_cond_req + UA_IHX_req
        
        # 9. 결과 반환
        return -COP, Total_UA, T_evap_K, T_cond_K

    except Exception as e:
        # CoolProp 오류, fsolve 수렴 실패 등
        # print(f"Error: {e}") # 디버깅용
        return -1e-9, 1e12, 0.0, 0.0
    

if __name__ == "__main__":
    
    # 1. 고정 변수 정의 (ihx_cycle_ntumethod.py 기준)
    fixed_params = {
        "ref": "R1233zd(E)",
        "Q_H_target": 200 * 1000,
        "T_source_in_K": 75.0 + 273.15,
        "m_dot_source": 1.51,
        "T_sink_in_K": 115.0 + 273.15, # T_sink 115 C
        "m_dot_sink": 10.0,            # m_dot_sink 10 kg/s
        "superheat_K": 5.0,
        "subcool_K": 2.0,
        "eta_comp": 0.75,
        "dp_ratio_gas": 0.05,
        "dp_ratio_liquid": 0.02
        # UA_IHX는 이제 입력 변수이므로 여기서 제외
    }
    
    # 2. 테스트할 변수
    test_T_evap_C = 70.0  # (T_source 75.0 보다 낮아야 함)
    test_T_cond_C = 120.0 # (T_sink 115.0 보다 높아야 함)
    test_epsilon_ihx = 0.5 # IHX 유용도 50%
    
    test_p_evap = PropsSI('P', 'T', test_T_evap_C + 273.15, 'Q', 1, fixed_params["ref"])
    test_p_cond = PropsSI('P', 'T', test_T_cond_C + 273.15, 'Q', 1, fixed_params["ref"])

    test_inputs = [test_p_evap, test_p_cond, test_epsilon_ihx]
    
    # 3. 함수 호출
    print(f"--- IHX (Epsilon={test_epsilon_ihx}) 적용 사이클 테스트 ---")
    print(f"    (T_evap={test_T_evap_C}°C, T_cond={test_T_cond_C}°C)")

    neg_COP, Total_UA, T_evap_K_out, T_cond_K_out = evaluate_ihx_cycle(test_inputs, fixed_params)
    
    # 4. 결과 출력
    actual_COP = -neg_COP
    
    print("\n--- 테스트 결과 ---")
    if actual_COP < 0.01:
        print("   > 계산 실패")
    else:
        print(f"  > T_evap: {T_evap_K_out - 273.15:.3f} °C")
        print(f"  > T_cond: {T_cond_K_out - 273.15:.3f} °C")
        print(f"  > 실제 COP: {actual_COP:.3f}")
        print(f"  > 필요 총 UA (Evap+Cond+IHX): {Total_UA:.2f} W/K")