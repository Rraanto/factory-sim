"""A differentiable production, inventory, and delivery model for a factory."""

from dataclasses import dataclass
from enum import StrEnum

import torch


class DeliveryStatus(StrEnum):
    """Outcome of a product delivery at a given period."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class FactoryParameters:
    """Physical and commercial inputs for a planning horizon.

    All vectors use product order; demand has shape ``(periods, products)``.
    Energy prices are expressed in currency per kWh.
    """

    demand: torch.Tensor
    initial_inventory: torch.Tensor
    production_capacity: torch.Tensor
    process_energy_per_unit: torch.Tensor
    tariff: torch.Tensor
    lines: int = 1
    base_load_kwh: float = 0.0
    activation_kwh: float = 0.0
    failure_penalty: float = 1.0


@dataclass
class SimulationResult:
    """Tensor outputs and user-facing delivery statuses from one simulation."""

    inventory: torch.Tensor
    delivered: torch.Tensor
    unmet_demand: torch.Tensor
    energy_kwh: torch.Tensor
    energy_cost: torch.Tensor
    failure_cost: torch.Tensor
    total_cost: torch.Tensor
    delivery_statuses: tuple[tuple[DeliveryStatus, ...], ...]

    @property
    def delivery_success_rate(self) -> torch.Tensor:
        """Fraction of deliveries fulfilled in full."""
        return (self.unmet_demand <= 1e-6).to(self.unmet_demand.dtype).mean()


class FactoryDigitalTwin:
    """Auto-differentiable digital twin for aggregate factory planning.

    ``production[t, p]`` is the planned number of units of product ``p`` made
    during period ``t``. Production becomes available before the period's
    delivery. The returned scalar costs retain PyTorch's computation graph.
    """

    def __init__(self, parameters: FactoryParameters):
        self.parameters = parameters
        self._validate_parameters()

    @property
    def horizon(self) -> int:
        return self.parameters.demand.shape[0]

    @property
    def products(self) -> int:
        return self.parameters.demand.shape[1]

    def _validate_parameters(self) -> None:
        p = self.parameters
        if p.demand.ndim != 2:
            raise ValueError("demand must have shape (periods, products)")
        if p.lines < 1:
            raise ValueError("lines must be at least 1")
        if p.initial_inventory.shape != (p.demand.shape[1],):
            raise ValueError("initial_inventory must have one value per product")
        for name, value in (
            ("production_capacity", p.production_capacity),
            ("process_energy_per_unit", p.process_energy_per_unit),
        ):
            if value.shape != (p.demand.shape[1],):
                raise ValueError(f"{name} must have one value per product")
        if p.tariff.shape != (p.demand.shape[0],):
            raise ValueError("tariff must have one value per period")

    def capacity_constraints(self, production: torch.Tensor) -> torch.Tensor:
        """Return remaining line capacity per period (non-negative is feasible)."""
        self._validate_production(production)
        line_days_used = (production / self.parameters.production_capacity).sum(dim=1)
        return self.parameters.lines - line_days_used

    def simulate(self, production: torch.Tensor) -> SimulationResult:
        """Run the twin and return differentiable costs plus discrete statuses."""
        self._validate_production(production)
        p = self.parameters
        inventory = p.initial_inventory
        inventories, delivered, unmet = [], [], []

        for period in range(self.horizon):
            available = inventory + production[period]
            shipped = torch.minimum(available, p.demand[period])
            shortage = p.demand[period] - shipped
            inventory = available - shipped
            inventories.append(inventory)
            delivered.append(shipped)
            unmet.append(shortage)

        inventory_history = torch.stack(inventories)
        delivered_tensor = torch.stack(delivered)
        unmet_tensor = torch.stack(unmet)
        line_days_used = (production / p.production_capacity).sum(dim=1)
        active_lines = p.lines * (1 - torch.exp(-line_days_used / p.lines))
        energy_kwh = (
            p.base_load_kwh
            + (production * p.process_energy_per_unit).sum(dim=1)
            + p.activation_kwh * active_lines
        )
        energy_cost = (energy_kwh * p.tariff).sum()
        failure_cost = p.failure_penalty * unmet_tensor.sum()
        statuses = tuple(
            tuple(
                DeliveryStatus.SUCCESS if value <= 1e-6 else DeliveryStatus.FAILURE
                for value in row.detach().cpu().tolist()
            )
            for row in unmet_tensor
        )
        return SimulationResult(
            inventory=inventory_history,
            delivered=delivered_tensor,
            unmet_demand=unmet_tensor,
            energy_kwh=energy_kwh,
            energy_cost=energy_cost,
            failure_cost=failure_cost,
            total_cost=energy_cost + failure_cost,
            delivery_statuses=statuses,
        )

    def objective(self, production: torch.Tensor, capacity_penalty: float = 0.0) -> torch.Tensor:
        """Return energy plus delivery-failure cost, optionally penalising overload."""
        result = self.simulate(production)
        overload = torch.relu(-self.capacity_constraints(production)).square().sum()
        return result.total_cost + capacity_penalty * overload

    def _validate_production(self, production: torch.Tensor) -> None:
        if production.shape != (self.horizon, self.products):
            raise ValueError(
                f"production must have shape ({self.horizon}, {self.products})"
            )
        if not production.is_floating_point():
            raise TypeError("production must be a floating-point tensor")
