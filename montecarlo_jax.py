import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, pmap, lax
import time
from functools import partial


# Enable float32 for faster computation on CPU
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platform_name', 'cpu')  # Explicitly use CPU
jax.config.update('jax_default_matmul_precision', 'float32')

@partial(jit, static_argnums=(1,))
def simulate_points_vectorized(key, num_points):
    """
    Vectorized version of point simulation optimized for CPU.
    Uses smaller batch sizes to prevent memory pressure.
    """
    points = random.uniform(key, shape=(num_points, 2))
    min_vals = jnp.min(points, axis=1)
    max_vals = jnp.max(points, axis=1)
    return jnp.sum(min_vals), jnp.sum(max_vals)

@partial(jit, static_argnums=(1, 2))
def run_batched_simulation(key, total_points, batch_size):
    """
    CPU-optimized batched simulation using efficient scanning.
    """
    num_batches = total_points // batch_size

    def scan_body(carry, idx):
        key, min_sum, max_sum = carry
        batch_key = random.fold_in(key, idx)  # More efficient than split
        batch_min_sum, batch_max_sum = simulate_points_vectorized(batch_key, batch_size)
        return (key, min_sum + batch_min_sum, max_sum + batch_max_sum), None

    (_, total_min_sum, total_max_sum), _ = lax.scan(
        scan_body,
        (key, 0.0, 0.0),
        jnp.arange(num_batches)
    )

    return total_min_sum, total_max_sum

def run_optimized_simulation(total_points, batch_size):
    """
    Run the optimized simulation with CPU-specific optimizations.
    Uses threading for parallel processing on CPU cores.
    """
    # Adjust batch size based on available CPU threads
    num_threads = jax.device_count()
    points_per_thread = total_points // num_threads
    points_per_thread = (points_per_thread // batch_size) * batch_size
    total_points = points_per_thread * num_threads

    # Generate thread keys
    thread_keys = random.split(random.PRNGKey(0), num_threads)

    start_time = time.time()

    # Run simulation on each thread
    results = []
    for i in range(num_threads):
        min_sum, max_sum = run_batched_simulation(thread_keys[i], points_per_thread, batch_size)
        results.append((min_sum, max_sum))

    # Sum results across threads
    total_min_sum = sum(r[0] for r in results)
    total_max_sum = sum(r[1] for r in results)

    expected_min = float(total_min_sum) / total_points
    expected_max = float(total_max_sum) / total_points

    elapsed_time = time.time() - start_time
    return expected_min, expected_max, elapsed_time

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPU-Optimized JAX simulations")
    parser.add_argument("-s", "--simulations", type=int, default=100_000_000,
                        help="Number of simulations to run (default: 100,000,000)")
    parser.add_argument("-b", "--batch-size", type=int, default=100_000,
                        help="Batch size for processing (default: 100,000)")
    args = parser.parse_args()

    # Pre-compile with a small batch
    _ = run_optimized_simulation(100000, 3256)

    print(f"JAX CPU threads available: {jax.device_count()}")

    total_points = args.simulations
    batch_size = args.batch_size

    print(f"\nRunning {total_points:,} optimized simulations...")
    expected_min, expected_max, elapsed_time = run_optimized_simulation(total_points, batch_size)

    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Expected value of minimum point: {expected_min:.6f}")
    print(f"Expected value of maximum point: {expected_max:.6f}")

    # Theoretical values for comparison
    theoretical_min, theoretical_max = 1/3, 2/3
    print(f"\nDifference from theoretical (min): {abs(expected_min - theoretical_min):.6f}")
    print(f"Difference from theoretical (max): {abs(expected_max - theoretical_max):.6f}")