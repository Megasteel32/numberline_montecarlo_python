import jax
import jax.numpy as jnp
from jax import random, vmap, jit, pmap, lax
import argparse
import time
from functools import partial
import numpy as np

# Enable float32 for faster computation
jax.config.update('jax_enable_x64', False)

@partial(jit, static_argnums=(1,))
def simulate_points(key, num_points):
    """Generate random points and compute min/max statistics using JAX."""
    points = random.uniform(key, shape=(num_points, 2))
    min_vals = jnp.minimum(points[:, 0], points[:, 1])
    max_vals = jnp.maximum(points[:, 0], points[:, 1])
    return jnp.sum(min_vals), jnp.sum(max_vals)

@partial(jit, static_argnums=(1, 2))
def run_batched_simulation(key, total_points, batch_size):
    """Run simulation in batches."""
    num_batches = total_points // batch_size

    def body_fun(i, carry):
        key, min_sum, max_sum = carry
        batch_key, next_key = random.split(key)
        batch_min_sum, batch_max_sum = simulate_points(batch_key, batch_size)
        return (next_key, min_sum + batch_min_sum, max_sum + batch_max_sum)

    _, total_min_sum, total_max_sum = lax.fori_loop(
        0, num_batches,
        body_fun,
        (key, 0.0, 0.0)
    )

    return total_min_sum, total_max_sum

def parallel_process(total_points, batch_size):
    """Process chunks of points across available devices."""
    num_devices = jax.device_count()
    points_per_device = total_points // num_devices

    # Create separate keys for each device
    keys = random.split(random.PRNGKey(0), num_devices)

    # Map the computation across devices
    device_fn = pmap(lambda key: run_batched_simulation(key, points_per_device, batch_size))
    device_results = device_fn(keys)

    # Sum results across devices
    total_min_sum = jnp.sum(device_results[0])
    total_max_sum = jnp.sum(device_results[1])

    return total_min_sum, total_max_sum

def run_simulation(total_points, batch_size):
    # Ensure total points is divisible by both batch size and device count
    num_devices = jax.device_count()
    points_per_device = total_points // num_devices
    points_per_device = (points_per_device // batch_size) * batch_size
    total_points = points_per_device * num_devices

    start_time = time.time()

    # Run simulation
    total_min_sum, total_max_sum = parallel_process(total_points, batch_size)

    # Calculate final statistics
    expected_min = float(total_min_sum) / total_points
    expected_max = float(total_max_sum) / total_points

    elapsed_time = time.time() - start_time
    return expected_min, expected_max, elapsed_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX simulations")
    parser.add_argument("-s", "--simulations", type=int, default=100_000_000,
                        help="Number of simulations to run (default: 100,000,000)")
    parser.add_argument("-b", "--batch-size", type=int, default=1_000_000,
                        help="Batch size for processing (default: 1,000,000)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable JAX profiling")
    args = parser.parse_args()

    if args.profile:
        from jax.profiler import start_trace, stop_trace
        start_trace('./trace')

    print(f"JAX devices available: {jax.devices()}")
    print(f"Running with float32 precision")

    total_points = args.simulations
    batch_size = args.batch_size

    print(f"\nRunning {total_points:,} simulations...")
    print(f"Batch size: {batch_size:,}")

    expected_min, expected_max, elapsed_time = run_simulation(total_points, batch_size)

    if args.profile:
        stop_trace()

    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Number of simulations: {total_points:,}")
    print(f"Expected value of minimum point: {expected_min:.6f}")
    print(f"Expected value of maximum point: {expected_max:.6f}")

    theoretical_min = 1/3
    theoretical_max = 2/3

    print(f"\nTheoretical expected value of minimum: {theoretical_min}")
    print(f"Theoretical expected value of maximum: {theoretical_max}")
    print(f"Difference from theoretical (min): {abs(expected_min - theoretical_min):.6f}")
    print(f"Difference from theoretical (max): {abs(expected_max - theoretical_max):.6f}")