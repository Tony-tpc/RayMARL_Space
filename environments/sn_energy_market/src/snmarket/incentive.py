"""DR uses signed request reduction; actual service cannot create extra DR credit."""

import math

from .contracts import DRIncentive, DRResult


def procure(orders, offer, dt_hours):
    orders = tuple(orders)
    signed = {o.consumer_id: (o.baseline_kw - o.request_kw if offer.event_active else 0.0) for o in orders}
    positive = {i: max(0.0, value) for i, value in signed.items()}
    gross = math.fsum(positive.values())
    net = max(0.0, math.fsum(signed.values()))
    procured = min(offer.dr_procurement_cap_kw, net)
    consumers = {}
    for identity in signed:
        paid = procured * positive[identity] / gross if gross > 0.0 else 0.0
        consumers[identity] = DRIncentive(identity, offer.event_id, signed[identity],
                                        positive[identity], paid, paid * offer.dr_incentive * dt_hours)
    payment = math.fsum(x.payment for x in consumers.values())
    credit = procured if offer.event_active and offer.dr_incentive > 0.0 else 0.0
    return DRResult(consumers, net, gross, procured, payment, credit)

