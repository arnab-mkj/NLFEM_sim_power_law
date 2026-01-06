import numpy as np
import matplotlib.pyplot as plt
import uuid

# Variant 8 Parameters
E = 100000  # Young's modulus (MPa)
B = 200     # Creep constant (MPa)
n = 13      # Creep exponent
A1, A2 = 6, 12  # Cross-sectional areas (mm^2)
L1, L2 = 20, 40  # Segment lengths (mm)
F_hat = 1200  # Final force (N)
t_tot = 20    # Total simulation time (s)
t1 = 1        # Time to reach F_hat (s)
dt = 0.001      # Time step (s, adjustable for convergence)
n_elements = 2  # Two elements
n_nodes = 3   # Three nodes (fixed at ends, force at middle)
gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]  # 2 Gauss points
gauss_weights = [1, 1]  # Gauss weights
max_iter = 100

# Material Routine: Computes stress, creep strain, and tangent stiffness
def material_routine(eps, eps_cr_m, dt, tol=1e-6, max_iter=50):
    # Initialize creep strain at current time step
    eps_cr = eps_cr_m
    sigma = E * (eps - eps_cr)
    
    # Newton-Raphson to solve for eps_cr^{m+1}
    for _ in range(max_iter):
        # Residual: R = eps_cr - eps_cr_m - dt * (abs(sigma)/B)^n * sign(sigma)
        sigma = E * (eps - eps_cr)
        R = eps_cr - eps_cr_m - dt * (abs(sigma)/B)**n * np.sign(sigma)
        if abs(R) < tol:
            break
        # Tangent: dR/deps_cr = 1 + dt * (n * abs(sigma)^(n-1) / B^n) * E
        dR_deps_cr = 1 + dt * (n * (abs(sigma)/B)**(n-1) / B) * E * np.sign(sigma)**2
        deps_cr = -R / dR_deps_cr
        eps_cr += deps_cr
    
    # Final stress and tangent stiffness
    sigma = E * (eps - eps_cr)
    C_t = E / (1 + dt * (n * (abs(sigma)/B)**(n-1) / B) * E)
    return sigma, eps_cr, C_t

# Element Routine: Computes internal forces and stiffness matrix
def element_routine(u_e, L, A, eps_cr_m, dt):
    F_int_e = np.zeros(2)  # Element internal force vector
    K_e = np.zeros((2, 2))  # Element stiffness matrix
    B = np.array([-1/L, 1/L])  # Strain-displacement matrix
    n_gauss = len(gauss_points)
    
    eps_cr = np.zeros(n_gauss)
    for i, (xi, w) in enumerate(zip(gauss_points, gauss_weights)):
        # Strain at Gauss point
        eps = np.dot(B, u_e)
        # Call material routine
        sigma, eps_cr[i], C_t = material_routine(eps, eps_cr_m[i], dt)
        # Internal force: F_int_e = ∫ (σ A B) dx = ∑ (σ A B) * (L/2) * w
        F_int_e += (sigma * A * B) * (L/2) * w
        # Stiffness: K_e = ∫ (B^T C_t B A) dx = ∑ (B^T C_t B A) * (L/2) * w
        K_e += np.outer(B, B) * C_t * A * (L/2) * w
        # Store creep strain at Gauss point (for simplicity, use average)
        # eps_cr_m = eps_cr
    
    return F_int_e, K_e, eps_cr

# Main Program
def main():
    # Initialize global arrays
    u = np.zeros(n_nodes)  # Global displacement vector
    eps_cr_m = np.zeros((n_elements, len(gauss_points)))  # Creep strain per element, Gauss point
    F_ext = np.zeros(n_nodes)  # External force vector
    times = np.arange(0, t_tot + dt, dt)
    results = {'time': [], 'u2': [], 'sigma1': [], 'sigma2': []}
    
    # Element connectivity and properties
    elements = [(0, 1, L1, A1), (1, 2, L2, A2)]  # (node1, node2, length, area)
    
    # Time loop
    for t in times:
        # Update external force
        F_ext[1] = F_hat * min(t/t1, 1)  # Linear ramp to F_hat at t1, then constant
        u_old = u.copy()
        iterations = 0
        converged = False
        # Newton-Raphson loop
        # for _ in range(max_iter):  # Max iterations
        while True:
            F_int = np.zeros(n_nodes)
            K = np.zeros((n_nodes, n_nodes))
            eps_cr_m_plus_1 = np.zeros_like(eps_cr_m)
            # Assemble global system
            for e, (n1, n2, L, A) in enumerate(elements):
                u_e = u[[n1, n2]]
                F_int_e, K_e, eps_cr_new = element_routine(u_e, L, A, eps_cr_m[e, :], dt)
                eps_cr_m_plus_1[e, :] = eps_cr_new
                # Assemble internal forces
                F_int[[n1, n2]] += F_int_e
                # Assemble stiffness matrix
                K[np.ix_([n1, n2], [n1, n2])] += K_e
            
            # Apply boundary conditions (u1 = u3 = 0)
            active_dof = [1]  # Only node 2 is free
            R = F_int - F_ext
            R[active_dof]
            K_red = K[np.ix_(active_dof, active_dof)]
            du = np.zeros(n_nodes)
            try:
                du[active_dof] = np.linalg.solve(K_red, -R[active_dof])
            except np.linalg.LinAlgError:
                print(f"Time {t:.2f} s: Linear system singular, skipping update")
                break
            
            iterations +=1
            
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
        
            # # Check for convergence (using norms is more robust)
            # residual_norm = np.linalg.norm(R[active_dof])
            # force_norm = np.linalg.norm(F_ext[active_dof])
            # displacement_norm = np.linalg.norm(du[active_dof])
            
            # if residual_norm < 0.005 * (force_norm + 1) and displacement_norm < 0.005 * u : # Check relative to external force
            #     print(f"Time {t:.2f} s: Converged in {iterations} iterations.")
            #     # CONVERGENCE! NOW we can update the state variable for the NEXT time step.
            #     eps_cr_m = eps_cr_m_plus_1.copy()
            #     break # Exit the NR while loop

            
        # Compute stresses for output
        sigma = []
        for e, (n1, n2, L, A) in enumerate(elements):
            u_e = u[[n1, n2]]
            eps = (u_e[1] - u_e[0]) / L
            sigma_e, _, _ = material_routine(eps, np.mean(eps_cr_m[e, :]), dt)
            sigma.append(sigma_e)
        
        # Store results
        results['time'].append(t)
        results['u2'].append(u[1])
        results['sigma1'].append(sigma[0])
        results['sigma2'].append(sigma[1])
    
    # Save results to file
    with open('results.txt', 'w') as f:
        f.write('Time(s) U2(mm) Sigma1(MPa) Sigma2(MPa)\n')
        for t, u2, s1, s2 in zip(results['time'], results['u2'], results['sigma1'], results['sigma2']):
            f.write(f'{t:.2f} {u2:.6f} {s1:.6f} {s2:.6f}\n')
    
    # Plot stresses vs. time
    plt.title((f'stress_vs_time_dt{dt}'))
    plt.plot(results['time'], results['sigma1'], label='Segment 1')
    plt.plot(results['time'], results['sigma2'], label='Segment 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Stress (MPa)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'stress_vs_time_dt{dt}.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()