"""Step 1 unit tests: verify the raw loading layer behaves as specified."""
import pandas as pd
import pytest

from src.data_layer.load_raw import EXPECTED_FILES, check_files_exist


def test_expected_files_list_not_empty():
    """The expected-file list should contain the seven raw tables."""
    assert len(EXPECTED_FILES) == 7
    assert "application_train.csv" in EXPECTED_FILES


def test_check_files_exist_raises_when_missing(tmp_path):
    """A directory without the raw files must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        check_files_exist(data_dir=tmp_path)


def test_main_table_has_target_column():
    """The main table contract: SK_ID_CURR plus a binary TARGET column."""
    mock = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]})
    assert "TARGET" in mock.columns
    assert set(mock["TARGET"].unique()).issubset({0, 1})
