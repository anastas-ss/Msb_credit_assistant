import os
import sqlite3
from datetime import datetime, timedelta

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "clients", "clients.sqlite")


def _connect():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _today():
    """«Сегодня» берём из метаданных БД (данные сгенерированы на эту дату),
    чтобы расчёты были воспроизводимы и совпадали с ожиданиями кейсов."""
    try:
        con = _connect()
        row = con.execute("select value from generation_metadata where key='today'").fetchone()
        con.close()
        if row:
            return datetime.strptime(row[0], "%Y-%m-%d").date()
    except Exception:
        pass
    return datetime.today().date()


def _resolve_access(authorized_client_id, requested_client_id):
    """Проверка доступа. Возвращает (target_id, error_response_or_None)."""
    if not authorized_client_id:
        return None, {"status": "need_auth", "data": None}
    target = requested_client_id or authorized_client_id
    if str(target).upper() != str(authorized_client_id).upper():
        return None, {"status": "access_denied", "data": None}
    return authorized_client_id, None


def _months_between(date_str, today):
    """Сколько полных месяцев прошло с date_str до today (грубо, по 30 дней)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (today - d).days // 30)
    except Exception:
        return None


def get_client_profile(authorized_client_id, requested_client_id=None):
    target, err = _resolve_access(authorized_client_id, requested_client_id)
    if err:
        return err
    con = _connect()
    row = con.execute(
        "select legal_form, industry, okved_main, region, annual_revenue, "
        "credit_score, employees_count, registration_date, has_account_in_bank, "
        "account_open_date, has_payroll_project, has_active_overdue, "
        "max_overdue_days_12m, current_debt_load from clients where client_id=?",
        (target,),
    ).fetchone()
    con.close()
    if not row:
        return {"status": "not_found", "data": None}
    today = _today()
    data = dict(row)
    data["client_id"] = target
    data["business_age_months"] = _months_between(row["registration_date"], today)
    data["account_age_months"] = _months_between(row["account_open_date"], today) if row["account_open_date"] else None
    return {"status": "ok", "data": data}


def get_client_applications(authorized_client_id, requested_client_id=None):
    target, err = _resolve_access(authorized_client_id, requested_client_id)
    if err:
        return err
    con = _connect()
    rows = con.execute(
        "select application_id, product_code, amount_requested, term_requested_months, "
        "application_date, status, decision, decision_reason_category "
        "from applications where client_id=? order by application_date desc",
        (target,),
    ).fetchall()
    con.close()
    apps = [dict(r) for r in rows]
    return {"status": "ok", "data": {"client_id": target, "applications": apps, "count": len(apps)}}


def get_client_credits(authorized_client_id, requested_client_id=None):
    target, err = _resolve_access(authorized_client_id, requested_client_id)
    if err:
        return err
    con = _connect()
    rows = con.execute(
        "select contract_id, product_name, product_code, principal_outstanding, interest_rate, "
        "term_months, months_passed, next_payment_date, next_payment_amount, has_overdue, "
        "overdue_days, overdue_amount, is_restructured, restructuring_count "
        "from credit_products where client_id=? order by contract_date desc",
        (target,),
    ).fetchall()
    con.close()
    credits = [dict(r) for r in rows]
    return {"status": "ok", "data": {"client_id": target, "credits": credits, "count": len(credits)}}


def get_application_status(authorized_client_id, requested_client_id=None):
    result = get_client_applications(authorized_client_id, requested_client_id)
    if result["status"] != "ok":
        return result
    apps = result["data"]["applications"]
    if not apps:
        return {"status": "not_found", "data": None}
    latest = apps[0]
    today = _today()
    try:
        submitted = datetime.strptime(latest["application_date"], "%Y-%m-%d").date()
        latest["days_since_submitted"] = (today - submitted).days
    except Exception:
        latest["days_since_submitted"] = None
    return {"status": "ok", "data": latest}


def _full_early_repayment(outstanding, rate, next_payment_date, today):
    """Оценка суммы полного досрочного погашения:
    остаток основного долга + проценты, накопленные с даты последнего платежа.
    Дату последнего платежа оцениваем как next_payment_date − 1 месяц (~30 дней).
    """
    if outstanding is None or rate is None:
        return None
    try:
        next_pay = datetime.strptime(next_payment_date, "%Y-%m-%d").date()
        last_pay = next_pay - timedelta(days=30)
        days_accrued = max(0, (today - last_pay).days)
    except Exception:
        days_accrued = 0
    accrued_interest = outstanding * (rate / 100.0) * days_accrued / 365.0
    return {
        "principal_outstanding": outstanding,
        "interest_rate": rate,
        "days_accrued": days_accrued,
        "accrued_interest": round(accrued_interest, 2),
        "total_to_repay": round(outstanding + accrued_interest, 2),
    }


def get_active_credit_summary(authorized_client_id, requested_client_id=None):
    result = get_client_credits(authorized_client_id, requested_client_id)
    if result["status"] != "ok":
        return result
    credits = result["data"]["credits"]
    if not credits:
        return {"status": "not_found", "data": None}
    today = _today()
    for c in credits:
        c["early_repayment"] = _full_early_repayment(
            c.get("principal_outstanding"), c.get("interest_rate"),
            c.get("next_payment_date"), today,
        )
    return {"status": "ok", "data": {"client_id": result["data"]["client_id"], "credits": credits}}


PRODUCT_NAMES = {
    "BUSINESS_OBOROT": "Бизнес-Оборот",
    "BUSINESS_RAZVITIE": "Бизнес-Развитие",
    "BUSINESS_START": "Бизнес-Старт",
    "BUSINESS_LIMIT": "Бизнес-Лимит",
    "BUSINESS_PEREZAGRUZKA": "Бизнес-Перезагрузка",
}


def product_name(code):
    """Человеческое название продукта по коду (если код незнаком - вернём как есть)."""
    return PRODUCT_NAMES.get(code, code)
