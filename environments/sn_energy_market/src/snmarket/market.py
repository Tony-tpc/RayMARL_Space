"""v9 necessary-first clearing: DER PAB, UC proportional, then DER rescue."""

import math

from .contracts import Allocation, LayerAllocation, MarketResult


def proportional(requests, capacity):
    total = math.fsum(requests.values())
    ratio = min(1.0, max(0.0, capacity) / total) if total > 0.0 else 0.0
    return {i: q * ratio for i, q in requests.items()}


def allocate_der(requests, bids, capacity):
    """Descending bid groups; each tied group shares proportionally."""
    fills = dict.fromkeys(requests, 0.0)
    remaining = max(0.0, capacity)
    for price in sorted(set(bids.values()), reverse=True):
        group = {i: requests[i] for i in sorted(requests) if bids[i] == price}
        assigned = proportional(group, remaining)
        fills.update(assigned)
        remaining = max(0.0, remaining - math.fsum(assigned.values()))
    return fills


def clear(orders, offer):
    """Clear validated orders; the engine checks conservation before committing."""
    orders = tuple(orders)
    by_id = {order.consumer_id: order for order in orders}
    if len(by_id) != len(orders):
        raise ValueError("Duplicate consumer orders")
    ids = sorted(by_id)
    bids = {i: by_id[i].der_bid_price for i in ids}
    der_left, uc_left = offer.der_capacity_kw, offer.uc_capacity_kw
    layers = []
    for layer in range(2):
        req = {i: (by_id[i].necessary_kw if layer == 0 else
                   max(0.0, by_id[i].request_kw - by_id[i].necessary_kw)) for i in ids}
        intent = {}
        for i in ids:
            prior_der = layers[0][i].der_kw if layer else 0.0
            intent[i] = min(req[i], max(0.0, by_id[i].der_fraction * by_id[i].request_kw - prior_der))
        first = allocate_der(intent, bids, der_left)
        der_left = max(0.0, der_left - math.fsum(first.values()))
        after_first = {i: max(0.0, req[i] - first[i]) for i in ids}
        uc = proportional(after_first, uc_left)
        uc_left = max(0.0, uc_left - math.fsum(uc.values()))
        after_uc = {i: max(0.0, after_first[i] - uc[i]) for i in ids}
        rescue = allocate_der(after_uc, bids, der_left)
        der_left = max(0.0, der_left - math.fsum(rescue.values()))
        layers.append({i: LayerAllocation(req[i], first[i], uc[i], rescue[i],
                                          max(0.0, after_uc[i] - rescue[i])) for i in ids})
    allocations = {i: Allocation(i, layers[0][i], layers[1][i]) for i in ids}
    der_total = math.fsum(a.der_kw for a in allocations.values())
    uc_total = math.fsum(a.uc_kw for a in allocations.values())
    return MarketResult(allocations, der_total, uc_total, der_left, uc_left)

