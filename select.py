from metrics.oof_metrics import save_oof_report

def pick_best_seed(model_name: str, seeds: list[int]):
    scores = {}
    reports = {}
    for s in seeds:
        rep = save_oof_report(model_name, s)
        if rep is None:
            continue
        scores[s] = rep["acc"]
        reports[s] = rep
    if not scores:
        return None, {}, {}
    best = max(scores, key=scores.get)
    return best, scores, reports

def build_class_weights_from_reports(best_reports: dict, alpha=1.0, eps=1e-3):
    """
    best_reports: {model_name: report_json}
    return class_weights: {model_name: [C]}
    """
    models = list(best_reports.keys())
    C = len(best_reports[models[0]]["f1"])
    # use f1 as class skill
    raw = {}
    for m in models:
        f1 = best_reports[m]["f1"]
        raw[m] = [(x + eps) ** alpha for x in f1]
    # normalize per-class
    class_weights = {m: [0.0]*C for m in models}
    for c in range(C):
        denom = sum(raw[m][c] for m in models)
        for m in models:
            class_weights[m][c] = raw[m][c] / (denom + 1e-12)
    return class_weights
