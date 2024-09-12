import numpy as np
from numba import jit
from multiprocessing import Pool, shared_memory
from tqdm import tqdm
import time
import argparse

@jit(nopython=True)
def simulate_points(num_simulations):
    points = np.random.random((num_simulations, 2))
    min_sum = np.sum(np.minimum(points[:, 0], points[:, 1]))
    max_sum = np.sum(np.maximum(points[:, 0], points[:, 1]))
    return min_sum, max_sum

def worker(args):
    num_simulations, shm_name, offset = args
    shm = shared_memory.SharedMemory(name=shm_name)
    result = np.ndarray((2,), dtype=np.float64, buffer=shm.buf[offset:offset+16])
    min_sum, max_sum = simulate_points(num_simulations)
    result[0] = min_sum
    result[1] = max_sum
    shm.close()
    return offset

def parallel_simulate(total_simulations, num_processes):
    simulations_per_process = total_simulations // num_processes

    shm = shared_memory.SharedMemory(create=True, size=num_processes * 16)
    result_array = np.ndarray((num_processes, 2), dtype=np.float64, buffer=shm.buf)

    pool = Pool(processes=num_processes)

    start_time = time.time()

    with tqdm(total=num_processes, desc="Simulating", unit="batch") as pbar:
        results = pool.imap_unordered(worker,
                                      [(simulations_per_process, shm.name, i*16) for i in range(num_processes)])

        for _ in results:
            pbar.update()

    pool.close()
    pool.join()

    total_min_sum = np.sum(result_array[:, 0])
    total_max_sum = np.sum(result_array[:, 1])

    shm.close()
    shm.unlink()

    expected_min = total_min_sum / total_simulations
    expected_max = total_max_sum / total_simulations

    return expected_min, expected_max, time.time() - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel simulations")
    parser.add_argument("-s", "--simulations", type=int, default=100_000_000,
                        help="Number of simulations to run (default: 100,000,000)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="Number of processes to use (default: 1)")
    args = parser.parse_args()

    total_simulations = args.simulations
    num_processes = args.processes

    print(f"Running {total_simulations:,} simulations using {num_processes} processes...")
    expected_min, expected_max, elapsed_time = parallel_simulate(total_simulations, num_processes)

    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Number of simulations: {total_simulations:,}")
    print(f"Number of processes: {num_processes}")
    print(f"Expected value of minimum point: {expected_min:.6f}")
    print(f"Expected value of maximum point: {expected_max:.6f}")

    theoretical_min = 1/3
    theoretical_max = 2/3

    print(f"\nTheoretical expected value of minimum: {theoretical_min}")
    print(f"Theoretical expected value of maximum: {theoretical_max}")
    print(f"Difference from theoretical (min): {abs(expected_min - theoretical_min):.6f}")
    print(f"Difference from theoretical (max): {abs(expected_max - theoretical_max):.6f}")