def _split_citation(c):
    """'01_credit_products.md#2.1.2' -> ('01_credit_products.md', '2.1.2')."""
    c = (c or "").strip()
    if "#" in c:
        f, a = c.split("#", 1)
        return f.strip(), a.strip()
    return c, ""


def _anchor_covers(a, b):
    """Якорь a покрывает b, если a - префикс b по сегментам ('2' покрывает '2.1.2')."""
    pa = [x for x in a.split(".") if x]
    pb = [x for x in b.split(".") if x]
    return pb[: len(pa)] == pa and len(pa) > 0


def source_hit(refs, sources):
    """Есть ли среди источников агента хотя бы один, совпадающий с эталонным.
    Совпадение - по файлу + по якорю (равны или один покрывает другой)."""
    for ref in refs or []:
        rf, ra = _split_citation(ref)
        for src in sources or []:
            sf, sa = _split_citation(src)
            if rf != sf:
                continue
            if not ra or not sa:
                return True
            if ra == sa or _anchor_covers(sa, ra) or _anchor_covers(ra, sa):
                return True
    return False


def _safe_ratio(hits, total):
    return round(hits / total, 3) if total else None


def compute_metrics(records):
    """records - список словарей с полями:
       category, expected, predicted, predicted_escalation,
       tool_status, is_transactional, refs, sources.
    """
    total = len(records)
    overall_hits = sum(1 for r in records if r["predicted"] == r["expected"])

    by_cat = {}
    cats = sorted(set(r["category"] for r in records))
    for cat in cats:
        items = [r for r in records if r["category"] == cat]
        hits = sum(1 for r in items if r["predicted"] == r["expected"])
        by_cat[cat] = {"accuracy": _safe_ratio(hits, len(items)), "n": len(items)}

    esc = [r for r in records if r["expected"] == "escalation"]
    esc_hits = sum(1 for r in esc if r["predicted"] == "escalation")

    rej = [r for r in records if r["expected"] == "rejection"]
    rej_hits = sum(1 for r in rej if r["predicted"] == "rejection")

    tx = [r for r in records if r["is_transactional"]]
    tool_hits = sum(1 for r in tx if r["tool_status"] == "ok")

    doc = [r for r in records if r["expected"] in ("info", "calculation") and r["refs"]]
    src_hits = sum(1 for r in doc if source_hit(r["refs"], r["sources"]))

    return {
        "overall_accuracy": _safe_ratio(overall_hits, total),
        "n_total": total,
        "accuracy_by_category": by_cat,
        "escalation_accuracy": _safe_ratio(esc_hits, len(esc)),
        "escalation_n": len(esc),
        "rejection_accuracy": _safe_ratio(rej_hits, len(rej)),
        "rejection_n": len(rej),
        "tool_success_rate": _safe_ratio(tool_hits, len(tx)),
        "tool_n": len(tx),
        "rag_source_hit_rate": _safe_ratio(src_hits, len(doc)),
        "rag_n": len(doc),
    }
