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


def client_exists(client_id):
    """True, если клиент с таким client_id есть в БД."""
    if not client_id:
        return False
    try:
        con = _connect()
        row = con.execute(
            "select 1 from clients where upper(client_id)=upper(?)", (str(client_id),)
        ).fetchone()
        con.close()
        return row is not None
    except Exception:
        return False


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


def _add_months(d, months):
    """Добавляет months месяцев к date без внешних зависимостей."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    month_lengths = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(d.day, month_lengths[month - 1])
    return d.replace(year=year, month=month, day=day)


def _adjust_next_payment(credit, today=None):
    """Ближайший БУДУЩИЙ платёж. Если сохранённая в БД дата уже прошла (бот запущен
    позже даты генерации данных), сдвигаем её на месячные периоды вперёд в рамках
    срока кредита - чтобы не называть прошедшую дату следующим платежом."""
    today = today or datetime.today().date()
    raw = credit.get("next_payment_date")
    if not raw:
        return None
    try:
        pay_date = datetime.strptime(raw, "%Y-%m-%d").date()
    except Exception:
        return None
    months_shifted = 0
    term_months = credit.get("term_months")
    months_passed = credit.get("months_passed") or 0
    max_extra_months = None
    if term_months is not None:
        max_extra_months = max(0, int(term_months) - int(months_passed))
    while pay_date < today:
        if max_extra_months is not None and months_shifted >= max_extra_months:
            return None
        months_shifted += 1
        pay_date = _add_months(pay_date, 1)
    return {"next_payment_date": pay_date.isoformat(), "months_shifted": months_shifted}


def _active_credits_from_result(result):
    """Активными считаем кредиты с остатком основного долга > 0 (явного status в БД нет)."""
    if result.get("status") != "ok":
        return []
    credits = result.get("data", {}).get("credits", []) or []
    return [c for c in credits if (c.get("principal_outstanding") or 0) > 0]


def get_credit_presence(authorized_client_id, requested_client_id=None):
    """Узкий tool для вопроса «есть ли у меня кредиты?»."""
    result = get_client_credits(authorized_client_id, requested_client_id)
    if result["status"] != "ok":
        return result
    active = _active_credits_from_result(result)
    return {"status": "ok", "data": {
        "client_id": result["data"]["client_id"],
        "has_active_credits": bool(active),
        "active_count": len(active),
        "credits": active,
    }}


def get_active_credit_balance(authorized_client_id, requested_client_id=None):
    """Узкий tool для вопросов про остаток/баланс долга."""
    result = get_client_credits(authorized_client_id, requested_client_id)
    if result["status"] != "ok":
        return result
    active = _active_credits_from_result(result)
    if not active:
        return {"status": "not_found", "data": None}
    total_outstanding = sum((c.get("principal_outstanding") or 0) for c in active)
    return {"status": "ok", "data": {
        "client_id": result["data"]["client_id"],
        "credits": active,
        "count": len(active),
        "total_principal_outstanding": total_outstanding,
    }}


def get_next_payment(authorized_client_id, requested_client_id=None):
    """Узкий tool для вопроса «когда следующий платёж?». При нескольких кредитах берём
    ближайшую дату; прошедшую дату из БД сдвигаем в будущее (_adjust_next_payment)."""
    result = get_client_credits(authorized_client_id, requested_client_id)
    if result["status"] != "ok":
        return result
    active = _active_credits_from_result(result)
    if not active:
        return {"status": "not_found", "data": None}

    def key(c):
        try:
            return datetime.strptime(c.get("next_payment_date"), "%Y-%m-%d").date()
        except Exception:
            return datetime.max.date()

    credit = sorted(active, key=key)[0]
    adjusted = _adjust_next_payment(credit)
    npd = adjusted["next_payment_date"] if adjusted else credit.get("next_payment_date")
    return {"status": "ok", "data": {
        "client_id": result["data"]["client_id"],
        "credit": credit,
        "next_payment_date": npd,
        "next_payment_amount": credit.get("next_payment_amount"),
    }}


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
