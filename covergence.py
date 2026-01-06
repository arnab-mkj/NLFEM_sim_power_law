import numpy as np
import matplotlib.pyplot as plt
import uuid

# Variant 8 Parameters
E = 100000  # Young's modulus (MPa)
B = 200     # Creep constant (MPa)
n = 13      # Creep exponent
A1, A2 = 6, 12  # Cross-sectional areas (mm^2)
L1, L2 = 20, 40  # Total segment lengths (mm)
F_hat = 1200  # Final force (N)
t_tot = 20    # Total simulation time (s)
t1 = 1        # Time to reach F_hat (s)
max_iter = 100  # Max iterations for global NR

# Generate element connectivity based on number of elements
def generate_elements(n_elements):
    elements = []
    n_nodes = n_elements + 1
    # Divide L1 and L2 into n_elements/2 each (assuming equal division for simplicity)
    len_per_element_1 = L1 / (n_elements // 2)
    len_per_element_2 = L2 / (n_elements // 2)
    for i in range(n_elements // 2):
        elements.append((i, i + 1, len_per_element_1, A1))
    for i in range(n_elements // 2):
        elements.append((n_elements // 2 + i, n_elements // 2 + i + 1, len_per_element_2, A2))
    return elements, n_nodes

# Material Routine: Computes stress, creep strain, and tangent stiffness
def material_routine(eps, eps_cr_m, dt, tol=1e-6, max_iter=50):
    eps_cr = eps_cr_m
    sigma = E * (eps - eps_cr)
    
    # for _ in range(max_iter):
    sigma = E * (eps - eps_cr)
    R = eps_cr - eps_cr_m - dt * (abs(sigma)/B)**n * np.sign(sigma)
    # if abs(R) < tol:
    #     print("here")
    #     break
    dR_deps_cr = 1 + dt * (n * (abs(sigma)/B)**(n-1) / B) * E * np.sign(sigma)**2
    deps_cr = -R / dR_deps_cr
    eps_cr += deps_cr
    
    sigma = E * (eps - eps_cr)
    C_t = E / (1 + dt * (n * (abs(sigma)/B)**(n-1) / B) * E)
    return sigma, eps_cr, C_t

# Element Routine: Computes internal forces and stiffness matrix
def element_routine(u_e, L, A, eps_cr_m, dt):
    F_int_e = np.zeros(2)
    K_e = np.zeros((2, 2))
    B = np.array([-1/L, 1/L])
    n_gauss = 2  # Using 2 Gauss points as defined
    
    eps_cr = np.zeros(n_gauss)
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    gauss_weights = [1, 1]
    for i, (xi, w) in enumerate(zip(gauss_points, gauss_weights)):
        eps = np.dot(B, u_e)
        sigma, eps_cr[i], C_t = material_routine(eps, eps_cr_m[i], dt)
        F_int_e += (sigma * A * B) * (L/2) * w
        K_e += np.outer(B, B) * C_t * A * (L/2) * w
    
    return F_int_e, K_e, eps_cr

# Main Program for a single run
def run_simulation(n_elements, dt):
    elements, n_nodes = generate_elements(n_elements)  # Unpack n_nodes
    u = np.zeros(n_nodes)
    eps_cr_m = np.zeros((n_elements, 2))  # 2 Gauss points
    F_ext = np.zeros(n_nodes)
    times = np.arange(0, t_tot + dt, dt)
    results = {'time': [], 'u2': []}
    
    for t in times:
        F_ext[1] = F_hat * min(t/t1, 1)
        iterations = 0
        converged = False
        
        while True:
            F_int = np.zeros(n_nodes)
            K = np.zeros((n_nodes, n_nodes))
            eps_cr_m_plus_1 = np.zeros_like(eps_cr_m)
            for e, (n1, n2, L, A) in enumerate(elements):
                u_e = u[[n1, n2]]
                F_int_e, K_e, eps_cr_new = element_routine(u_e, L, A, eps_cr_m[e, :], dt)
                eps_cr_m_plus_1[e, :] = eps_cr_new
                F_int[[n1, n2]] += F_int_e
                K[np.ix_([n1, n2], [n1, n2])] += K_e
            
            active_dof = [1]
            R = F_int - F_ext
            K_red = K[np.ix_(active_dof, active_dof)]
            du = np.zeros(n_nodes)
            try:
                du[active_dof] = np.linalg.solve(K_red, -R[active_dof])
            except np.linalg.LinAlgError:
                print(f"Time {t:.2f} s: Linear system singular, skipping update")
                break
            
            iterations += 1
            u += du
            # Check convergence
            norm_R = np.max(np.abs(R[active_dof]))
            norm_F_int = np.max(np.abs(F_int[active_dof]))
            norm_du = np.max(np.abs(du[active_dof]))
            norm_u = np.max(np.abs(u[active_dof]))

            # Define the tolerances from the assignment
            force_tolerance = 0.005
            disp_tolerance = 0.005

            # Check for convergence. Add a small number (e.g., 1e-9) 
            # to prevent multiplication by zero if forces or displacements are zero at the start.
            force_check_passed = norm_R < force_tolerance * (norm_F_int + 1e-9)
            disp_check_passed = norm_du < disp_tolerance * (norm_u + 1e-9)
            
            if force_check_passed and disp_check_passed:
                print(f"Time {t:.2f} s: Converged in {iterations} iterations.")
                # CONVERGENCE! NOW we can update the state variable for the NEXT time step.
                eps_cr_m = eps_cr_m_plus_1.copy()
                break # Exit the NR while loop
        
        # if not converged:# and iterations < 1000:
        #     print(f"Time {t:.2f} s: Did not converge in {iterations} iterations")
        
        results['time'].append(t)
        results['u2'].append(u[1])
    
    return u[1]  # Return u2 at the last time step

# Convergence Studies
def convergence_study():
    # Reference solution (finest mesh and smallest dt)
    ref_n_elements = 2
    ref_dt = 0.1
    ref_u2 = run_simulation(ref_n_elements, ref_dt)
    print(f"Reference u2 at t=20s with n_elements={ref_n_elements}, dt={ref_dt}: {ref_u2:.6f}")
    
    # h-Convergence (vary n_elements)
    n_elements_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    # h_errors = []
    # h_sizes = []
    u2_values_h = []
    for n_e in n_elements_values:
        u2 = run_simulation(n_e, 0.1)  # Fixed dt for h-convergence
        u2_values_h.append(u2)
        error = abs(u2 - ref_u2)
        # h_errors.append(error)
        # h_sizes.append((L1 + L2) / n_e)  # Average element size
        # print(f"n_elements={n_e}, u2={u2:.6f}, Error={error:.6f}")
    
    # dt-Convergence (vary dt)
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    # dt_errors = []
    u2_values_dt = []
    for dt in dt_values:
        u2 = run_simulation(2, dt)  # Fixed n_elements for dt-convergence
        u2_values_dt.append(u2)
        # error = abs(u2 - ref_u2)
        # dt_errors.append(error)
        # print(f"dt={dt}, u2={u2:.6f}, Error={error:.6f}")
    
    # Plots
    plt.figure(figsize=(12, 5))
    
    # h-Convergence Plot
    plt.subplot(1, 2, 1)
    plt.plot(n_elements_values, u2_values_h, 'bo-', label='displacement u2 vs h')
    plt.xlabel('number of elements')
    plt.ylabel('Displacement u2 (mm)')
    plt.title('h-Convergence Study')
    plt.grid(True)
    plt.legend()
    
    # dt-Convergence Plot
    plt.subplot(1, 2, 2)
    plt.plot(dt_values, u2_values_dt, 'ro-', label='displacement u2 vs dt')
    plt.xlabel('Time Step dt (s)')
    plt.ylabel('Displacement u2 (mm)')
    plt.title('dt-Convergence Study')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_study.png')
    plt.show()

if __name__ == '__main__':
    convergence_study()