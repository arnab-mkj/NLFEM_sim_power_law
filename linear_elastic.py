import numpy as np
import matplotlib.pyplot as plt

# Params for variant 8
E = 100000 # youngs modulus(MPa)
A1, A2 = 6, 12 # cross-sectional areas(mm^2)
L1, L2 = 20, 40 # segment lengths(mm)
F_hat = 1200 # final force(N)
t_tot = 1 # simulate upto t=1s for linear-elastic case
t1 = 1 #time to reach F_hat(s)
dt = 0.1 # time step(s)
n_elements = 2 # two elements
n_nodes = 3 # three nodes
gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)] # 2 gauss points
gauss_weights = [1, 1] # gauss weights

# Analytical
def analytical_solution(t):
    k1 = E * A1/L1 # stiffness of segment 1(N/mm)
    k2 = E * A2/L2 # stiffness of segment 2
    # k_eff = 1/((1/k1) + (1/k2)) # effective stiffness series(N/mm)
    k_eff = k1 + k2 # effective stiffness parallel
    F = F_hat * min(t/t1, 1) # force at time t(N)
    u2 = F/k_eff # displacement at node 2(mm)
    sigma1 = E * (u2 / L1) # stress in segment 1 (MPa)
    sigma2 = E * (-u2 / L2) # stress im segment 2(Mpa)
    # sigma1 = F/A1
    # sigma2 = -F/A2
    return u2, sigma1, sigma2

# material routine for linear-elestic (eps_cr = 0)
def material_routine(eps, eps_P = 0.0):
    
    sigma = E * eps # stress: sigma = E * (eps - eps_r), eps_cr = 0
    C_t = E # Tangent stiffness: C_t = E for linear-elastic
    return sigma, C_t

# element routine
def element_routine(u_e, L, A):
    F_int_e = np.zeros(2) # element interal force vector
    K_e = np.zeros((2, 2)) # element stiffness matrix
    B = np.array([-1/L, 1/L]) # strain-displacement matrix
    
    for x_i, w in zip(gauss_points, gauss_weights):
        eps = np.dot(B, u_e) # strain
        sigma, C_t = material_routine(eps) # call material routine
        detJ = L / 2  # Jacobian for [0, L] mapping
        F_int_e += (sigma * A * B) * w * detJ # internal force
        K_e += np.outer(B, B) * C_t * A * w * detJ # stiffness matrix
        
    return F_int_e, K_e

# main
def main():
    u = np.zeros(n_nodes) # displacement vector
    F_ext = np.zeros(n_nodes) #exernal force vector
    times = np.arange(0, t_tot + dt, dt)
    results = {'time': [], 'u2': [], 'sigma1': [], 'sigma2': []} 
    analytical_results = {'time': [], 'u2': [], 'sigma1': [], 'sigma2': []}
    
    # element connectivity
    elements = [(0, 1, L1, A1), (1, 2, L2, A2)] # (node1, node2, length, area)
    
    # print intermediate results
    print("Intermdeiate Results: ")
    print(f"B_1 = [-1/{L1}, 1/{L1}] = {[-1/L1, 1/L1]}")
    print(f"B_2 = [-1/{L2}, 1/{L2}] = {[-1/L2, 1/L2]}")
    print(f"K_1 = {E * A1 / L1} * [[1, -1], [-1, 1]]")
    print(f"K_1 = {E * A2 / L2} * [[1, -1], [-1, 1]]")
    
    
    # time loop
    for t in times:
        F_ext[1] = F_hat * min(t/t1, 1) #external force
        # print(F_ext[1])
        # F_ext[1] = 0
        iterations = 0
        
        # newton rhapson loop
        for _ in range(1): # max iterations (expect 1 for linear case)
            F_int = np.zeros(n_nodes)
            K = np.zeros((n_nodes, n_nodes))
            iterations +=1
            
            #assemble for global system
            for e, (n1, n2, L, A) in enumerate(elements):
                u_e = u[[n1, n2]]
                F_int_e, K_e = element_routine(u_e, L, A)
                # Ensure consistent force direction
                
                F_int[[n1, n2]] += F_int_e
                
                K[np.ix_([n1, n2], [n1, n2])] += K_e
                
            #apply boundary conditions (u1 = u3 = 0)
            active_dof = [1] # node 2
            R = F_int - F_ext
            K_red = K[np.ix_(active_dof, active_dof)]
            du = np.zeros(n_nodes)
            du[active_dof] = np.linalg.solve(K_red, -R[active_dof])
            # du[active_dof] = -R[active_dof] / K_red
            
            #check convergence
            if np.max(np.abs(R)) < (0.005 * np.max(np.abs(F_int))) and \
                np.max(np.abs(du)) < (0.005 * np.max(np.abs(u))):
                print(f"Time {t:.2f} s: Converged in {iterations} iterations")
                break
            u += du
            
        # compute stresses
        sigma = []
        for e, (n1, n2, L, A) in enumerate(elements):
            u_e = u[[n1, n2]]
            # print(u_e)
            eps = (u_e[1] - u_e[0]) / L
            # print(eps)
            # print(e)
            sigma_e, _ = material_routine(eps)
            sigma.append(sigma_e)
            
        # store FEM results 
        results['time'].append(t)
        results['u2'].append(u[1])
        results['sigma1'].append(sigma[0])
        results['sigma2'].append(sigma[1])
        # print (results['sigma1'])
        
        #compute and store analytical results
        u2_anal, sigma1_anal, sigma2_anal = analytical_solution(t)
        analytical_results['time'].append(t)
        analytical_results['u2'].append(u2_anal)
        analytical_results['sigma1'].append(sigma1_anal)
        analytical_results['sigma2'].append(sigma2_anal)
        
    # Save results to file
    with open('linear_elastic_results.txt', 'w') as f:
        f.write('Time(s) U2_FEM(mm) U2_Anal(mm) Sigma1_FEM(MPa) Sigma1_Anal(MPa) Sigma2_FEM(MPa) Sigma2_Anal(MPa)\n')
        for t, u2, s1, s2, u2_a, s1_a, s2_a in zip(
                                results['time'], 
                                results['u2'], 
                                results['sigma1'], 
                                results['sigma2'],
                                analytical_results['u2'], 
                                analytical_results['sigma1'], 
                                analytical_results['sigma2']):
            f.write(f'{t:.2f} {u2:.6f} {u2_a:.6f} {s1:.6f} {s1_a:.6f} {s2:.6f} {s2_a:.6f}\n')
            
    #Plotting
    plt.figure(figsize=(12, 8))
    
    #Displacement plot
    # Displacement plot
    plt.subplot(3, 1, 1)
    plt.plot(results['time'], results['u2'], 'b-', label='FEM u2')
    plt.plot(analytical_results['time'], analytical_results['u2'], 'r--', label='Analytical u2')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement u2 (mm)')
    plt.legend()
    plt.grid(True)
    
    # Stress in Segment 1
    plt.subplot(3, 1, 2)
    plt.plot(results['time'], results['sigma1'], 'b-', label='FEM sigma1')
    plt.plot(analytical_results['time'], analytical_results['sigma1'], 'r--', label='Analytical sigma1')
    plt.xlabel('Time (s)')
    plt.ylabel('Stress sigma1 (MPa)')
    plt.legend()
    plt.grid(True)
    
    # Stress in Segment 2
    plt.subplot(3, 1, 3)
    plt.plot(results['time'], results['sigma2'], 'b-', label='FEM sigma2')
    plt.plot(analytical_results['time'], analytical_results['sigma2'], 'r--', label='Analytical sigma2')
    plt.xlabel('Time (s)')
    plt.ylabel('Stress sigma2 (MPa)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig('linear_elastic_results.png')
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    main()
            
            
        
        