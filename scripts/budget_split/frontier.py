"""Numerical implementation used by the reported experiment."""



def pareto(points):
    # All points in a panel share video/query denominators. Integer objectives
    # avoid false dominance from the last bits of percentages or mean costs.
    for p in points:
        p['pareto'] = not any(q['total_stored_vectors'] <= p['total_stored_vectors'] and
                             q['recall_hit_sum'] >= p['recall_hit_sum'] and
                             (q['total_stored_vectors'] < p['total_stored_vectors'] or
                              q['recall_hit_sum'] > p['recall_hit_sum']) for q in points)
    return points
