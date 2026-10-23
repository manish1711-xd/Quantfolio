import sqlite3
import pytest

from src.equity_research.history import (
    save_research_record,
    load_research_history,
    delete_research_record,
    init_history_db,
)


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test_research.db")
    init_history_db(db_path)
    return db_path


def test_save_and_load_history(test_db):
    rec_id = save_research_record(
        ticker="TCS.NS",
        company_name="Tata Consultancy Services",
        price=3500.0,
        intrinsic_value=3900.0,
        recommendation="Buy",
        summary="Solid operating performance and healthy ROE.",
        db_path=test_db,
    )
    assert rec_id > 0

    history = load_research_history(db_path=test_db)
    assert len(history) == 1
    assert history[0]["ticker"] == "TCS.NS"
    assert history[0]["price"] == 3500.0
    assert history[0]["intrinsic_value"] == 3900.0
    assert history[0]["recommendation"] == "Buy"


def test_delete_history(test_db):
    rec_id = save_research_record(
        ticker="INFY.NS",
        company_name="Infosys",
        price=1500.0,
        intrinsic_value=1600.0,
        recommendation="Hold",
        summary="Neutral near-term outlook.",
        db_path=test_db,
    )
    delete_research_record(rec_id, db_path=test_db)
    history = load_research_history(db_path=test_db)
    assert len(history) == 0
