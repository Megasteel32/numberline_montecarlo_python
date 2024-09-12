import numpy as np
from numba import njit, prange, set_num_threads
import argparse
import time

# Define a structured array dtype for our points
point_dtype = np.dtype([('x', np.float64), ('y', np.float64)])

@njit
def xoroshiro128p_next(state):
    """Xoroshiro128+ random number generator."""
    result = (state[0] + state[1]) & 0xFFFFFFFFFFFFFFFF
    s1 = state[1] ^ state[0]
    state[0] = ((state[0] << 24) | (state[0] >> 40)) ^ s1 ^ (s1 << 16)
    state[1] = (s1 << 37) | (s1 >> 27)
    return result

@njit
def xoroshiro128p_uniform_float64(state):
    """Generate a uniform float64 in [0, 1) using Xoroshiro128+."""
    return (xoroshiro128p_next(state) >> 11) * (1.0 / 9007199254740992.0)

@njit(parallel=True)
def simulate_points(num_simulations, chunk_size):
    total_min_sum = 0.0
    total_max_sum = 0.0

    for _ in prange(num_simulations // chunk_size):
        local_state = np.array([np.random.randint(1, 2**32), np.random.randint(1, 2**32)], dtype=np.uint64)
        points = np.empty(chunk_size, dtype=point_dtype)

        for i in range(chunk_size):
            points[i]['x'] = xoroshiro128p_uniform_float64(local_state)
            points[i]['y'] = xoroshiro128p_uniform_float64(local_state)

        min_vals = np.minimum(points['x'], points['y'])
        max_vals = np.maximum(points['x'], points['y'])

        total_min_sum += np.sum(min_vals)
        total_max_sum += np.sum(max_vals)

    return total_min_sum, total_max_sum

def run_simulation(total_simulations, num_threads, chunk_size):
    set_num_threads(num_threads)
    start_time = time.time()

    total_min_sum, total_max_sum = simulate_points(total_simulations, chunk_size)

    expected_min = total_min_sum / total_simulations
    expected_max = total_max_sum / total_simulations

    elapsed_time = time.time() - start_time

    return expected_min, expected_max, elapsed_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel simulations")
    parser.add_argument("-s", "--simulations", type=int, default=100_000_000,
                        help="Number of simulations to run (default: 100,000,000)")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of threads to use (default: 1)")
    parser.add_argument("-c", "--chunk-size", type=int, default=1000,
                        help="Chunk size for batched processing (default: 1000)")
    args = parser.parse_args()

    total_simulations = args.simulations
    num_threads = args.threads
    chunk_size = args.chunk_size

    print(f"Running {total_simulations:,} simulations using {num_threads} threads...")
    print(f"Chunk size: {chunk_size}")
    expected_min, expected_max, elapsed_time = run_simulation(total_simulations, num_threads, chunk_size)

    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Number of simulations: {total_simulations:,}")
    print(f"Number of threads: {num_threads}")
    print(f"Expected value of minimum point: {expected_min:.6f}")
    print(f"Expected value of maximum point: {expected_max:.6f}")

    theoretical_min = 1/3
    theoretical_max = 2/3

    print(f"\nTheoretical expected value of minimum: {theoretical_min}")
    print(f"Theoretical expected value of maximum: {theoretical_max}")
    print(f"Difference from theoretical (min): {abs(expected_min - theoretical_min):.6f}")
    print(f"Difference from theoretical (max): {abs(expected_max - theoretical_max):.6f}")