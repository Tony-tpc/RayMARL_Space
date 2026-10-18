"""v9 immutable contracts: power kW, energy kWh, payment yuan."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Phase(StrEnum):
    UC_TURN = "UC_TURN"
    CONSUMER_TURN = "CONSUMER_TURN"
    TERMINATED = "TERMINATED"


@dataclass(frozen=True, kw_only=True)
class UCAction:
    grid_import_offer_kw: float
    der_planned_curtailment_kw: float
    der_to_ess_kw: float
    grid_to_ess_kw: float
    ess_discharge_offer_kw: float
    dr_procurement_cap_kw: float

    @property
    def charge_kw(self):
        return self.der_to_ess_kw + self.grid_to_ess_kw


@dataclass(frozen=True, kw_only=True)
class ConsumerAction:
    curtail_kw: float
    signed_shift_kw: float
    der_preference_fraction: float
    der_bid_per_kwh: float


@dataclass(frozen=True)
class ShiftTask:
    source_period: int
    deadline_period: int
    remaining_kwh: float
    task_id: str | None = None


@dataclass(frozen=True)
class ConsumerState:
    consumer_id: str
    monthly_uc_energy_kwh: float = 0.0
    tasks: tuple[ShiftTask, ...] = ()

    @property
    def backlog_kwh(self):
        return sum(task.remaining_kwh for task in self.tasks)


@dataclass(frozen=True)
class MarketState:
    episode_id: str
    period_id: int
    timestamp: datetime
    ess_energy_kwh: float
    consumers: tuple[ConsumerState, ...]
    last_totals: tuple[float, ...] = (0.0,) * 6
    initial_ess_energy_kwh: float = 0.0


@dataclass(frozen=True)
class PublicOffer:
    action: UCAction
    der_capacity_kw: float
    uc_capacity_kw: float
    der_tariff: float
    dr_incentive: float
    event_active: bool
    event_id: str
    period_id: int
    rule_version: str
    discharge_available_kw: float

    @property
    def dr_procurement_cap_kw(self):
        return self.action.dr_procurement_cap_kw

@dataclass(frozen=True)
class LocalBounds:
    cut_max_kw: float
    due_kw: float
    due_kwh: float
    backlog_kwh: float
    backlog_limit_kwh: float
    baseline_kw: float
    rigid_kw: float
    flex_kw: float
    dt_hours: float
    allow_new_shift: bool
    der_bid_min: float
    der_bid_max: float
    uc_tariff: float

    def shift_interval(self, cut_kw):
        low = -min(self.flex_kw - cut_kw, self.backlog_kwh / self.dt_hours)
        if self.due_kw > 0:
            return low, -self.due_kw
        high = min(self.baseline_kw - self.rigid_kw - cut_kw,
                   self.flex_kw - cut_kw,
                   (self.backlog_limit_kwh - self.backlog_kwh) / self.dt_hours)
        return low, high if self.allow_new_shift else min(high, 0.0)


@dataclass(frozen=True)
class MarketOrder:
    consumer_id: str
    consumer_index: int
    action: ConsumerAction
    baseline_kw: float
    rigid_kw: float
    flex_kw: float
    native_kw: float
    request_kw: float
    necessary_kw: float
    due_kw: float
    due_kwh: float
    uc_tariff: float

    @property
    def der_bid_price(self):
        return self.action.der_bid_per_kwh

    @property
    def der_fraction(self):
        return self.action.der_preference_fraction


@dataclass(frozen=True)
class LayerAllocation:
    request_kw: float
    der_first_kw: float
    uc_kw: float
    der_rescue_kw: float
    unfilled_kw: float

    @property
    def der_kw(self):
        return self.der_first_kw + self.der_rescue_kw


@dataclass(frozen=True)
class Allocation:
    consumer_id: str
    necessary: LayerAllocation
    early: LayerAllocation

    @property
    def der_kw(self):
        return self.necessary.der_kw + self.early.der_kw

    @property
    def uc_kw(self):
        return self.necessary.uc_kw + self.early.uc_kw

    @property
    def served_kw(self):
        return self.der_kw + self.uc_kw

    @property
    def unfilled_kw(self):
        return self.necessary.unfilled_kw + self.early.unfilled_kw


@dataclass(frozen=True)
class MarketResult:
    allocations: dict[str, Allocation]
    der_total_kw: float
    uc_total_kw: float
    der_remaining_kw: float
    uc_remaining_kw: float


@dataclass(frozen=True)
class PhysicalExecution:
    grid_buy_kw: float
    der_generation_kw: float
    der_planned_curtailment_kw: float
    der_unsold_kw: float
    der_spill_kw: float
    der_to_ess_kw: float
    grid_to_ess_kw: float
    ess_discharge_kw: float
    grid_offer_unused_kw: float
    ess_discharge_unused_kw: float
    ess_energy_after_kwh: float
    supply_served_kw: float
    power_residual_kw: float
    energy_residual_kwh: float = 0.0

    @property
    def ess_charge_kw(self):
        return self.der_to_ess_kw + self.grid_to_ess_kw

@dataclass(frozen=True)
class LoadService:
    consumer_id: str
    native_served_kw: float
    native_lost_kw: float
    repay_actual_kw: float
    repay_unfilled_kw: float
    backlog_after_kwh: float
    overdue_after_kwh: float
    due_capacity_shortfall_kwh: float
    tasks_after: tuple[ShiftTask, ...]
    request_residual_kw: float
    original_load_residual_kw: float
    queue_residual_kwh: float


@dataclass(frozen=True)
class DRIncentive:
    consumer_id: str
    event_id: str
    signed_dr_kw: float
    positive_dr_kw: float
    paid_dr_kw: float
    payment: float


@dataclass(frozen=True)
class DRResult:
    consumers: dict[str, DRIncentive]
    net_dr_kw: float
    positive_dr_kw: float
    procured_kw: float
    total_payment: float
    supply_credit_kw: float


@dataclass(frozen=True)
class ConsumerLedger:
    energy_payment: float
    der_payment: float
    uc_payment: float
    dr_income: float
    compensation_income: float
    net_bill: float
    adjustment_cost: float
    native_loss_cost: float
    waiting_cost: float
    overdue_cost: float
    terminal_pending_cost: float
    total_cost: float


@dataclass(frozen=True)
class Ledger:
    consumers: dict[str, ConsumerLedger]
    energy_revenue: float
    dr_payment: float
    compensation_payment: float
    grid_cost: float
    der_operating_cost: float
    external_cost: float
    cash_profit: float
    ess_cost: float
    curtailment_cost: float
    book_profit: float
    supply_reference_kw: float
    announced_capacity_kw: float
    dr_credit_kw: float
    supply_shortfall_kw: float
    supply_shortfall_cost: float
    due_cost: float
    uc_service_cost: float
    dr_value: float
    terminal_energy_value: float
    terminal_pending_cost: float
    uc_reward: float
    cash_residual: float


@dataclass(frozen=True)
class ExecutionRecord:
    transition_id: str
    episode_id: str
    period_id: int
    timestamp: datetime
    mechanism_version: str
    input_signature: str
    uc_action: UCAction
    consumer_actions: dict[str, ConsumerAction]
    orders: tuple[MarketOrder, ...]
    offer: PublicOffer
    market: MarketResult
    physical: PhysicalExecution
    services: dict[str, LoadService]
    dr: DRResult
    ledger: Ledger
    rewards: dict[str, float]
    monthly_closing_energy_kwh: dict[str, float]
    month_rolled: bool
    terminated: bool
    truncated: bool = False
    decision_state: MarketState | None = None
    uc_observation: tuple[float, ...] = ()
    consumer_observations: dict[str, tuple[float, ...]] = field(default_factory=dict)
    local_bounds: dict[str, LocalBounds] = field(default_factory=dict)
    supply_diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StateDelta:
    before: MarketState
    after: MarketState
    record: ExecutionRecord
    payment_keys: tuple[tuple[str, int, str, str], ...]


@dataclass(frozen=True)
class EngineSnapshot:
    compatibility_signature: str
    state: MarketState
    phase: Phase
    pending_plan: UCAction | None
    pending_offer: PublicOffer | None
    pending_raw_uc: tuple[float, ...] | None
    records: dict[str, ExecutionRecord]
    payment_keys: frozenset[tuple[str, int, str, str]]
    rng_state: dict = field(default_factory=dict)

