import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from CoolProp.CoolProp import PropsSI
try:
    from nhx import evaluate_nhx_cycle
except ImportError:
    print("="*50)
    print("Error: Could not find 'nhx.py'.")
    print("="*50)
    exit()

T_source_in = 75.0 

class nhxProblem(Problem):

    def __init__(self, fixed_params):
        
        ref = fixed_params["ref"]
        

        self.T_evap_min_C = T_source_in - 15
        self.T_evap_max_C = T_source_in - 5

        self.T_cond_min_C = 110 + 1
        T_cond_max_C = 150.0          # arbitrary upper limit
        
        # --- Search range for optimization variables (pressure, in Pa ---
        p_evap_min_Pa = PropsSI('P', 'T', self.T_evap_min_C + 273.15, 'Q', 1, ref)
        p_evap_max_Pa = PropsSI('P', 'T', self.T_evap_max_C + 273.15, 'Q', 1, ref)
        
        p_cond_min_Pa = PropsSI('P', 'T', self.T_cond_min_C + 273.15, 'Q', 1, ref)
        p_cond_max_Pa = PropsSI('P', 'T', T_cond_max_C + 273.15, 'Q', 1, ref)
        
        xl = np.array([p_evap_min_Pa, p_cond_min_Pa])
        xu = np.array([p_evap_max_Pa, p_cond_max_Pa])
        
        # Initialize the Pymoo Problem class
        super().__init__(n_var=2,       # 2 Variables (P_evap, P_cond)
                         n_obj=2,       # 2 Targets (COP, UA)
                         n_constr=3,    # 3 Constraints (T_evap 2개, T_cond 1개)
                         xl=xl,         # Var. lower limit (Pa)
                         xu=xu)         # Var. upper limit (Pa)
        
        self.fixed_params = fixed_params

    def _evaluate(self, x, out, *args, **kwargs):
        
        results_F = [] # Objective results ([neg_COP, Total_UA])
        results_G = [] # Constraint results ([g1, g2, g3])

        for individual_inputs in x:
            # individual_inputs are [p_evap, p_cond]
            
            neg_COP, Total_UA, T_evap_K, T_cond_K = \
                evaluate_nhx_cycle(individual_inputs, self.fixed_params)
            
            # [Objectives 1, 2]
            results_F.append([neg_COP, Total_UA])

            # [Constraints] (form: g(x) <= 0)            
            if T_evap_K == 0.0 or T_cond_K == 0.0:
                # If failed (penalty)
                g1 = 1e6
                g2 = 1e6
                g3 = 1e6
            else:
                # If successful
                T_evap_C = T_evap_K - 273.15
                T_cond_C = T_cond_K - 273.15
            
                # ex) T_evap >= 60.0  ->  60.0 - T_evap <= 0
                g1 = self.T_evap_min_C - T_evap_C
                
                # ex) T_evap <= 74.0  ->  T_evap - 74.0 <= 0
                g2 = T_evap_C - self.T_evap_max_C
                
                # ex) T_cond >= 101.0 -> 101.0 - T_cond <= 0
                g3 = self.T_cond_min_C - T_cond_C

            results_G.append([g1, g2, g3])

        # Final results as numpy arrays
        out["F"] = np.array(results_F)
        out["G"] = np.array(results_G)


# ===================================================================
# 🚀 Optimization
# ===================================================================

if __name__ == "__main__":

    # 1. Fixed parameters
    fixed_params = {
        "ref": "R1233zd(E)",
        "Q_H_target": 200 * 1000,           # 200 kW
        "T_source_in_K": T_source_in + 273.15,    
        "m_dot_source": 1.51,            
        "T_sink_in_K": 110.0 + 273.15,      
        "m_dot_sink": 4.0,             
        "superheat_K": 5.0,
        "subcool_K": 2.0,                      
        "eta_comp": 0.75,                   
        "dp_ratio_gas": 0.05,               
        "dp_ratio_liquid": 0.02             
    }
    
    # 2. Create problem instance 
    problem = nhxProblem(fixed_params=fixed_params)
    
    # 3. Define optimization algorithm (NSGA-II)
    algorithm = NSGA2(
        pop_size=100,
        eliminate_duplicates=True
    )
    
    # 4. Set termination condition
    termination = ('n_gen', 200)
    
    # 5. Run Optimization
    print("="*50)
    print(f"Optimization Start (NHX Model)")
    print(f"Ref: {fixed_params['ref']}, T_source: {fixed_params['T_source_in_K']-273.15} C, T_sink: {fixed_params['T_sink_in_K']-273.15} C")
    print(f"Algorithm: {algorithm.__class__.__name__}, Population: {algorithm.pop_size}, Generations: {termination[1]}")
    print(f"Constraints: T_evap ({problem.T_evap_min_C}~{problem.T_evap_max_C} C), T_cond (>= {problem.T_cond_min_C} C)")
    print("="*50)
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True,
                   save_history=True)
                   
    print("="*50)
    print("Optimization Finished!")
    print("="*50)

    # 6. Extract results
    variables = res.X  # optimal variables [P_evap, P_cond]
    objectives = res.F # optimal objectives [neg_COP, Total_UA]
    
    # 7. Print results
    print("\n--- Optimal Solutions (Pareto Front) ---")
    print(f"{'Sol #':>5} | {'P_evap (Pa)':>10} | {'P_cond (Pa)':>10} | {'T_evap (C)':>10} | {'T_cond (C)':>10} | {'COP':>8} | {'Total UA (W/K)':>15}")
    print("-" * 55)
    
    ref_str = fixed_params["ref"]
    
    if variables is None or objectives is None:
        print("... No valid solutions found (res.X is None)")
        print("... Check constraint settings.")
        valid_solutions_found = False
    else:
        results_list = []
        for i in range(len(variables)):
            p_evap, p_cond = variables[i, 0], variables[i, 1]
            cop, ua = -objectives[i, 0], objectives[i, 1]
            
            if cop <= 0.01 or ua >= 1e8: continue
            
            try:
                t_evap_c = PropsSI('T', 'P', p_evap, 'Q', 1, ref_str) - 273.15
                t_cond_c = PropsSI('T', 'P', p_cond, 'Q', 1, ref_str) - 273.15
                results_list.append([p_evap, p_cond, t_evap_c, t_cond_c, cop, ua])
            except ValueError:
                continue 

        if not results_list:
            print("... No valid solutions found")
            valid_solutions_found = False
        else:
            df_results = pd.DataFrame(results_list, columns=['P_evap','P_cond','T_evap_C', 'T_cond_C', 'COP', 'Total_UA_W_K'])
            print(df_results.to_string(float_format="%.3f", index=False))
            valid_solutions_found = True


    # --- 8. Select the best solution based on weighted score ---
    if valid_solutions_found:
        print("\n--- Weighted Score Analysis (Finding Best Solution) ---")
        
        # Define weights (COP 80%, UA 20%)
        w_cop = 0.8
        w_ua = 0.2
        print(f"  Weights: COP = {w_cop*100}%, Total_UA = {w_ua*100}%")

        # 1. Normalize (Min-Max Scaling)
        # COP: Maximize (0~1)
        df_results['COP_norm'] = (df_results['COP'] - df_results['COP'].min()) / \
                                 (df_results['COP'].max() - df_results['COP'].min())
        
        # UA: Minimize (1~0) -> (1 - 정규화)
        df_results['UA_norm_inv'] = 1.0 - (df_results['Total_UA_W_K'] - df_results['Total_UA_W_K'].min()) / \
                                      (df_results['Total_UA_W_K'].max() - df_results['Total_UA_W_K'].min())

        # 2. Compute weighted score
        df_results['Score'] = (w_cop * df_results['COP_norm']) + (w_ua * df_results['UA_norm_inv'])

        # 3. Select best solution
        best_solution = df_results.loc[df_results['Score'].idxmax()]
        
        print(f"\n  [Best Solution based on weights] {T_source_in:.2f}")
        print(best_solution.to_string(float_format="%.3f"))

    # 9. Visualize Pareto front
    if valid_solutions_found:
        plt.figure(figsize=(10, 6))
        
        # All Pareto solutions
        plt.scatter(df_results['Total_UA_W_K'], df_results['COP'], 
                    facecolors='none', edgecolors='blue', label='Pareto Front')
        
        # Weighted best solution
        plt.scatter(best_solution['Total_UA_W_K'], best_solution['COP'], 
                    color='red', s=100, marker='*', zorder=10, 
                    label=f'Best (Score: {best_solution["Score"]:.3f})')
        
        plt.title(f"Pareto Front (IHX 3-Var) - {fixed_params['ref']}")
        plt.xlabel("Total Required UA (Evap + Cond + IHX) [W/K]")
        plt.ylabel("COP (Performance) [-]")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("\n[Warning] No valid Pareto front solutions found — plot not generated.")