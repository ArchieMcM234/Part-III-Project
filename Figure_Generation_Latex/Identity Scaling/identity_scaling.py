import sys 
sys.path.append("/Users/archie/documents/year4/project/code")
import numpy as np
import multiprocessing as mp
from functools import partial

import spin_tools.hamiltonians as hamiltonians
from spin_tools.gates import *
import spin_tools.quantum_tools as qt
from spin_tools.floquet_tools import ccd_floquet

# Define function to process each detuning for CCD method
def process_ccd_detuning(detuning, natural_freq, driving_freq, rabi_freq, phi_0, epsilon_m, phase_freq, theta_m, evaluation_time, evaluation_points, tol, fid_1):
    # Calculate the effective the ccd time evolution unitaries
    times, Us_driving_frame = qt.calculate_unitaries(1, evaluation_time, evaluation_points, hamiltonians.ccd_rwa,
                        driving_freq=driving_freq,
                        natural_freq=natural_freq-detuning, 
                        rabi_freq=rabi_freq, 
                        phi_0=phi_0,
                        epsilon_m=epsilon_m,
                        phase_freq=phase_freq,
                        theta_m=theta_m,
                        atol=tol,
                        rtol=tol)
    
    ccd_floquet_freq = np.sign(detuning)*2*ccd_floquet(natural_freq-detuning, driving_freq, rabi_freq, phi_0, epsilon_m, phase_freq, theta_m)[4]
    transformed_Us = qt.phase_boost_unitaries(Us_driving_frame, ccd_floquet_freq, times)
    
    # Calculate fidelity for qubit 2 (identity gate)
    fid_2 = qt.calculate_fidelities(identity, transformed_Us, 2)
    
    # Calculate combined fidelity
    fid_combined = (np.abs((6*fid_1-2)*(6*fid_2-2))+4)/20
    
    return fid_1, fid_2, fid_combined

# Define function to process each detuning for Rabi method
def process_rabi_detuning(detuning, natural_freq, driving_freq, rabi_freq, evaluation_time, evaluation_points, tol, fid_1):
    times, Us_driving_frame = qt.calculate_unitaries(
        1, 
        evaluation_time, 
        evaluation_points, 
        hamiltonians.rabi_rwa, 
        natural_freq=natural_freq-detuning, 
        driving_freq=driving_freq, 
        rabi_freq=rabi_freq, 
        atol=tol, 
        rtol=tol
    )

    effective_rabi_freq = np.sign(detuning)*np.sqrt(rabi_freq**2 + detuning**2)
    Us_effective_rabi = qt.phase_boost_unitaries(Us_driving_frame, effective_rabi_freq, times)
    fid_2 = qt.calculate_fidelities(identity, Us_effective_rabi, 2)

    fid_combined = (np.abs((6*fid_1-2)*(6*fid_2-2))+4)/20
    
    return fid_1, fid_2, fid_combined

# Main code
if __name__ == "__main__":
    natural_freq = 10  # 10ghz   
    driving_freq = 10
    rabi_freq = 0.005

    # CCD parameters
    phi_0, epsilon_m, phase_freq, theta_m = 0, rabi_freq/4, rabi_freq, 0

    # Integration parameters
    tol = 10**-5
    evaluation_points = 2000
    evaluation_time = 4/rabi_freq

    # iteration parameters
    detunings = np.linspace(0, 100*rabi_freq, 100)

    # Calculate fidelity for qubit 1 (CCD method)
    times, Us_driving_frame = qt.calculate_unitaries(1, evaluation_time, evaluation_points, hamiltonians.ccd_rwa,
                                driving_freq=driving_freq,
                                natural_freq=natural_freq, 
                                rabi_freq=rabi_freq, 
                                phi_0=phi_0,
                                epsilon_m=epsilon_m,
                                phase_freq=phase_freq,
                                theta_m=theta_m,
                                atol=tol,
                                rtol=tol)
    fid_1_ccd = qt.calculate_fidelities(identity, Us_driving_frame, 2)

    # Use multiprocessing for CCD method
    process_ccd_partial = partial(
        process_ccd_detuning,
        natural_freq=natural_freq,
        driving_freq=driving_freq,
        rabi_freq=rabi_freq,
        phi_0=phi_0,
        epsilon_m=epsilon_m,
        phase_freq=phase_freq,
        theta_m=theta_m,
        evaluation_time=evaluation_time,
        evaluation_points=evaluation_points,
        tol=tol,
        fid_1=fid_1_ccd
    )
    
    # Create a pool of workers
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    
    # Execute calculations in parallel
    ccd_results = pool.map(process_ccd_partial, detunings)
    
    # Unpack results
    fids_1_ccd, fids_2_ccd, fids_combined_ccd = zip(*ccd_results)
    
    # Extract the final time step fidelity for each detuning
    final_fids_combined_ccd = [fid[-1] for fid in fids_combined_ccd]
    
    # Find combined infidelity
    infidelities_combined_ccd = 1-np.array(final_fids_combined_ccd)
    
    # Normalize detunings by Rabi frequency
    detunings_normalized = detunings / rabi_freq

    # Reset for Rabi method
    tol = 10**-5
    

    # Calculate fidelity for qubit 1 (Rabi method)
    times, Us_driving_frame = qt.calculate_unitaries(
            1, 
            evaluation_time, 
            evaluation_points, 
            hamiltonians.rabi_rwa, 
            natural_freq=natural_freq, 
            driving_freq=driving_freq, 
            rabi_freq=rabi_freq, 
            atol=tol, 
            rtol=tol
        )
    fid_1_rabi = qt.calculate_fidelities(identity, Us_driving_frame, 2)

    # Use multiprocessing for Rabi method
    process_rabi_partial = partial(
        process_rabi_detuning,
        natural_freq=natural_freq,
        driving_freq=driving_freq,
        rabi_freq=rabi_freq,
        evaluation_time=evaluation_time,
        evaluation_points=evaluation_points,
        tol=tol,
        fid_1=fid_1_rabi
    )
    
    # Execute calculations in parallel
    rabi_results = pool.map(process_rabi_partial, detunings)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Unpack results
    fids_1_rabi, fids_2_rabi, fids_combined_rabi = zip(*rabi_results)
    
    # Extract the final time step fidelity for each detuning
    final_fids_combined_rabi = [fid[-1] for fid in fids_combined_rabi]
    
    infidelities_combined_rabi = 1-np.array(final_fids_combined_rabi)

    import pandas as pd

    # Create a DataFrame with the data
    data = {
        "detunings_normalized": detunings_normalized,
        "infidelities_combined_ccd": infidelities_combined_ccd,
        "infidelities_combined_rabi": infidelities_combined_rabi
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_path = '/Users/archie/documents/year4/project/test.csv'
    df.to_csv(csv_path, index=False)
    print(f"Fidelities and normalized detunings saved successfully to {csv_path}.")
