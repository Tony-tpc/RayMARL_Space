"""Independent v9 equation oracle: no production imports or environment calls.

The input dictionaries deliberately do not use snmarket contracts. This keeps
the comparison independent of production action decoding and data plumbing.
All power is kW, energy is kWh, time is h, and monetary outputs are yuan.
"""

import math


def due_energy(tasks, period):
    return math.fsum(task[3] for task in tasks if task[2] <= period)


def local_bounds(person, *, period, deadline, horizon, dt):
    """Return cut upper, and a function of cut returning signed shift bounds."""
    pending = math.fsum(task[3] for task in person.get("tasks", ()))
    due = min(due_energy(person.get("tasks", ()), period) / dt, person["flex"])
    cut_upper = min(person["base"] - person["rigid"], person["flex"] - due)

    def shift_bounds(cut):
        lower = -min(person["flex"] - cut, pending / dt)
        if due > 0:
            upper = -due
        else:
            upper = min(person["base"] - person["rigid"] - cut,
                        person["flex"] - cut, (person["backlog_limit"] - pending) / dt)
            if horizon is not None and period + deadline > horizon - 1:
                upper = 0.0
        return lower, upper

    return due, cut_upper, shift_bounds


def _allocate(requests, available, bids=None):
    """Sort exact prices and independently allocate each tie group pro rata."""
    result = [0.0 for _ in requests]
    prices = sorted(set(bids), reverse=True) if bids is not None else [None]
    for price in prices:
        group = [i for i in range(len(requests)) if bids is None or bids[i] == price]
        demand = math.fsum(requests[i] for i in group)
        fraction = min(1.0, available / demand) if demand > 0 else 0.0
        for i in group:
            result[i] = requests[i] * fraction
        available -= math.fsum(result[i] for i in group)
        available = max(0.0, available)
    return result


def _coefficient(rates, key, i, default=0.0):
    value = rates.get(key, default)
    return float(value[i] if isinstance(value, (list, tuple)) else value)


def evaluate(people, uc, *, der, grid_limit, energy, energy_min=0.0, energy_max=10.0,
             charge_max=25.0, discharge_max=25.0, eta_c=0.95, eta_d=0.95,
             dt=0.25, period=0, deadline=4, horizon=None, event=True,
             dr_price=0.5, dr_limit=40.0, grid_price=0.5, rates=None,
             terminal=False, initial_energy=None, month_rollover=False):
    """Compute a complete physical period directly from the v9 formulas.

    people: dictionaries with base, rigid, flex, backlog_limit, tariff, bid,
    cut, shift, rho; optional tasks=(id, creation, deadline, remaining_kwh), M.
    uc: six physical quantities in the document's order.
    rates: der_operation, throughput, spill, dr_value, adequacy, uc_due,
    compensation, loss, discomfort, waiting, late, uc_terminal, terminal,
    terminal_energy, epsilon, external. Individual rates can be lists.
    """
    rates = {} if rates is None else rates
    offer, curtail, ch_der, ch_pg, dis_offer, dr_cap = map(float, uc)
    tol = 1e-9
    assert all(math.isfinite(v) for v in uc)
    assert min(uc) >= -tol
    assert offer <= grid_limit + tol and ch_pg <= offer + tol
    assert curtail + ch_der <= der + tol
    assert ch_der + ch_pg <= min(charge_max, (energy_max - energy) / (eta_c * dt)) + tol
    assert dis_offer <= min(discharge_max, eta_d * (energy - energy_min) / dt) + tol
    assert (ch_der + ch_pg) * dis_offer <= tol
    assert dr_cap <= float(event) * dr_limit + tol
    der_cap, uc_cap = der - curtail - ch_der, offer + dis_offer - ch_pg
    native, request, necessary, dues, bids, intentions = [], [], [], [], [], []
    for person in people:
        due, cut_upper, bounds = local_bounds(person, period=period, deadline=deadline,
                                               horizon=horizon, dt=dt)
        assert -tol <= person["cut"] <= cut_upper + tol
        lower, upper = bounds(person["cut"])
        assert lower - tol <= person["shift"] <= upper + tol
        assert 0 <= person["rho"] <= 1
        p_native = person["base"] - person["cut"] - max(person["shift"], 0.0)
        p_request = person["base"] - person["cut"] - person["shift"]
        native.append(p_native)
        request.append(p_request)
        necessary.append(p_native + due)
        dues.append(due)
        bids.append(person["bid"])
        intentions.append(person["rho"] * p_request)
    count = len(people)
    total_der, total_uc = [0.0] * count, [0.0] * count
    der_left, uc_left = der_cap, uc_cap
    layers = []
    for layer in range(2):
        demands = necessary if layer == 0 else [request[i] - necessary[i] for i in range(count)]
        desired = [min(demands[i], max(0.0, intentions[i] - total_der[i])) for i in range(count)]
        initial = _allocate(desired, der_left, bids)
        der_left -= math.fsum(initial)
        remaining = [max(0.0, demands[i] - initial[i]) for i in range(count)]
        uc_alloc = _allocate(remaining, uc_left)
        uc_left -= math.fsum(uc_alloc)
        remaining = [max(0.0, remaining[i] - uc_alloc[i]) for i in range(count)]
        rescue = _allocate(remaining, max(0.0, der_left), bids)
        der_left -= math.fsum(rescue)
        for i in range(count):
            total_der[i] += initial[i] + rescue[i]
            total_uc[i] += uc_alloc[i]
        layers.append({"der_initial": initial, "uc": uc_alloc, "der_rescue": rescue,
                       "gap": [max(0.0, remaining[i] - rescue[i]) for i in range(count)]})
    served = [total_der[i] + total_uc[i] for i in range(count)]
    sold_uc, sold_der = math.fsum(total_uc), math.fsum(total_der)
    dis = min(dis_offer, sold_uc)
    grid = sold_uc + ch_pg - dis
    spill = der - ch_der - sold_der
    after_energy = energy + eta_c * (ch_der + ch_pg) * dt - dis * dt / eta_d
    signed_dr = [float(event) * (people[i]["base"] - request[i]) for i in range(count)]
    gross = math.fsum(max(v, 0.0) for v in signed_dr)
    net = max(0.0, math.fsum(signed_dr))
    paid = min(dr_cap, net)
    compensation_total = dr_price * paid * dt
    consumers = []
    for i, person in enumerate(people):
        loss = max(0.0, native[i] - served[i])
        repay = max(0.0, served[i] - native[i])
        left = repay * dt
        next_tasks = []
        for task in sorted(person.get("tasks", ()), key=lambda task: (task[2], task[1])):
            amount = max(0.0, task[3] - left)
            # Recognize an effectively complete subtraction, not a fixed energy
            # floor: an untouched genuinely small task must remain pending.
            if left > 0 and math.isclose(left, task[3], rel_tol=8e-15, abs_tol=0.0):
                amount = 0.0
            left = max(0.0, left - task[3])
            if amount > 0:
                next_tasks.append((*task[:3], amount))
        if person["shift"] > 0:
            next_tasks.append((f"new:{period}:{i}", period, period + deadline, person["shift"] * dt))
        backlog = math.fsum(task[3] for task in next_tasks)
        overdue = max(0.0, due_energy(person.get("tasks", ()), period) - repay * dt)
        dr_cash = compensation_total * max(signed_dr[i], 0.0) / gross if gross else 0.0
        energy_cash = (person["bid"] * total_der[i] + person["tariff"] * total_uc[i]) * dt
        loss_cash = _coefficient(rates, "compensation", i) * loss * dt
        bill = energy_cash - dr_cash - loss_cash
        flex = (_coefficient(rates, "discomfort", i)
                * (person["cut"] ** 2 + max(person["shift"], 0.0) ** 2)
                / max(person["base"], rates.get("epsilon", 1e-6)) * dt)
        service = (_coefficient(rates, "loss", i) * loss * dt
                   + _coefficient(rates, "waiting", i) * backlog * dt
                   + _coefficient(rates, "late", i) * overdue * dt)
        term = -_coefficient(rates, "terminal", i) * backlog if terminal else 0.0
        consumers.append({"der": total_der[i], "uc": total_uc[i], "serve": served[i],
                          "native": native[i], "request": request[i], "necessary": necessary[i],
                          "lost": loss, "repay": repay, "backlog": backlog, "due_after": overdue,
                          "tasks": tuple(next_tasks), "signed_dr": signed_dr[i], "dr_cash": dr_cash,
                          "energy_cash": energy_cash, "loss_cash": loss_cash, "bill": bill,
                          "flex_cost": flex, "service_cost": service, "terminal": term,
                          "reward": -bill - flex - service + term,
                          "monthly": 0.0 if month_rollover else person.get("M", 0.0) + total_uc[i] * dt})
    grid_cash = grid_price * grid * dt
    der_cash = rates.get("der_operation", 0.0) * (der - spill) * dt
    cash = math.fsum(p["bill"] for p in consumers) - grid_cash - der_cash - rates.get("external", 0.0)
    throughput_cost = rates.get("throughput", 0.0) * (ch_der + ch_pg + dis) * dt
    spill_cost = rates.get("spill", 0.0) * spill * dt
    credit = paid if event and dr_price > 0 else 0.0
    reference = math.fsum(p["base"] for p in people) + math.fsum(dues)
    gap = max(0.0, reference - der_cap - uc_cap - credit)
    adequacy = rates.get("adequacy", 0.0) * gap * dt
    due_cost = rates.get("uc_due", 0.0) * math.fsum(p["due_after"] for p in consumers)
    value = rates.get("dr_value", 0.0) * net * dt
    terminal_uc = 0.0
    if terminal:
        initial_energy = energy if initial_energy is None else initial_energy
        terminal_uc = (rates.get("terminal_energy", 0.0) * (after_energy - initial_energy)
                       - rates.get("uc_terminal", 0.0) * math.fsum(p["backlog"] for p in consumers))
    return {"people": consumers, "layers": layers, "der_capacity": der_cap, "uc_capacity": uc_cap,
            "der_sold": sold_der, "uc_sold": sold_uc, "grid": grid, "discharge": dis,
            "energy_after": after_energy, "spill": spill, "unsold": spill - curtail,
            "gross_dr": gross, "net_dr": net, "paid_dr": paid, "credit_dr": credit,
            "cash": cash, "grid_cash": grid_cash, "der_cash": der_cash,
            "throughput_cost": throughput_cost, "spill_cost": spill_cost,
            "reference": reference, "adequacy_gap": gap, "adequacy_cost": adequacy,
            "due_cost": due_cost, "dr_value": value, "terminal": terminal_uc,
            "book": cash - throughput_cost - spill_cost,
            "reward": cash - throughput_cost - spill_cost - adequacy - due_cost + value + terminal_uc}
