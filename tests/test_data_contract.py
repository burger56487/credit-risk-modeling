"""Step 2 (revised) tests for the fixed field contract and its guards.

These are pure Python checks: they need no database and no real CSV.
"""
import pytest

from src.data_layer import ingest_to_db as ing
from src.data_layer.load_raw import EXPECTED_FILES


def test_pipeline_requirement_is_derived_from_the_contract():
    assert ing.required_files() == ("application_train.csv", "bureau.csv")
    # The dataset inventory is a different list, on purpose.
    assert len(EXPECTED_FILES) == 7
    assert set(ing.required_files()) <= set(EXPECTED_FILES)


def test_contract_columns_and_keys_are_explicit():
    application = ing.contract_for("application_train.csv")
    bureau = ing.contract_for("bureau.csv")

    assert application.table == "application_train"
    assert len(application.columns) == 12
    assert application.primary_key == ("sk_id_curr",)
    assert application.staging_table == "stg_application_train"

    assert bureau.table == "bureau"
    assert len(bureau.columns) == 7
    assert bureau.primary_key == ("sk_id_bureau",)

    with pytest.raises(KeyError, match="不在当前装载契约内"):
        ing.contract_for("bureau_balance.csv")


def test_contract_columns_are_declared_in_the_schema_expectations():
    """The contract and the structure check must not drift apart."""
    for item in ing.PIPELINE_CONTRACT:
        declared = ing.EXPECTED_TYPES[item.table]
        assert set(item.columns) == set(declared)
        assert ing.EXPECTED_PRIMARY_KEYS[item.table] == item.primary_key


def test_headers_are_normalised_and_collisions_are_fatal():
    assert ing.normalize_headers([" SK_ID_CURR ", "TARGET"]) == [
        "sk_id_curr",
        "target",
    ]

    with pytest.raises(ValueError, match="规范化后出现重复列名"):
        ing.normalize_headers(["SK_ID_CURR", "sk_id_curr"])

    with pytest.raises(ValueError, match="空白列名"):
        ing.normalize_headers(["SK_ID_CURR", "  "])


def test_extra_columns_are_registered_not_silently_dropped():
    contract = ing.contract_for("bureau.csv")

    headers = list(contract.columns) + [
        "CREDIT_TYPE",
        "AMT_CREDIT_MAX_OVERDUE",
    ]
    loaded, extra = ing.resolve_columns(contract, headers)

    assert loaded == list(contract.columns)
    assert extra == ["credit_type", "amt_credit_max_overdue"]


def test_missing_required_column_is_fatal():
    contract = ing.contract_for("application_train.csv")
    headers = [c for c in contract.columns if c != "target"]

    with pytest.raises(ValueError, match="缺少装载契约要求的列"):
        ing.resolve_columns(contract, headers)


def test_connection_configuration_has_no_default(monkeypatch):
    monkeypatch.delenv(ing.DB_URL_ENV, raising=False)

    with pytest.raises(RuntimeError, match="未配置数据库连接"):
        ing.get_engine()

    with pytest.raises(ValueError, match="空白字符串"):
        ing.get_engine("   ")

    monkeypatch.setenv(ing.DB_URL_ENV, "sqlite:///:memory:")
    engine = ing.get_engine()

    assert engine.dialect.name == "sqlite"


def test_module_no_longer_ships_a_connection_string():
    source = (
        __import__("pathlib").Path(ing.__file__).read_text(encoding="utf-8")
    )

    assert "creditrisk:creditrisk" not in source
    assert "DEFAULT_DB_URL" not in source


def test_derived_scripts_are_named_and_present():
    for name in ing.DERIVED_SCRIPTS:
        digest = ing.script_digest(name)
        assert len(digest) == 64


def test_contract_dataclass_rejects_inconsistent_keys():
    with pytest.raises(ValueError, match="主键列必须属于装载列"):
        ing.FileContract(
            file_name="x.csv",
            table="x",
            columns=("a",),
            primary_key=("b",),
            staging_table="stg_x",
        )
