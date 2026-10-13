"""Small, reproducible scenarios used by the optimisation notebooks."""

import torch

from modules.digital_twin import FactoryParameters


def example_parameters(dtype: torch.dtype = torch.float64) -> FactoryParameters:
    """Return a two-product, seven-day scenario inspired by the source factory."""
    return FactoryParameters(
        demand=torch.tensor(
            [[30, 10], [10, 25], [20, 20], [35, 5], [15, 30], [20, 15], [25, 20]],
            dtype=dtype,
        ),
        initial_inventory=torch.tensor([10, 5], dtype=dtype),
        production_capacity=torch.tensor([45, 35], dtype=dtype),
        process_energy_per_unit=torch.tensor([0.16, 0.22], dtype=dtype),
        tariff=torch.tensor([0.12, 0.10, 0.08, 0.16, 0.21, 0.18, 0.11], dtype=dtype),
        lines=2,
        base_load_kwh=8.0,
        activation_kwh=3.0,
        failure_penalty=2.5,
    )
