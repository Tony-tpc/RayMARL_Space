import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from experiments.sn_energy_market import ausgrid
from experiments.sn_energy_market.ausgrid import (
    IDENTITY_COLUMNS,
    INTERVAL_COLUMNS,
    TIME_POLICY,
    audit_households,
    meter_arrays,
    read_meter_csv,
    select_households,
)


def meter_frame(*, start="2010-07-01", days=2, customers=(1,), cl=True):
    rows = []
    for customer in customers:
        for date in pd.date_range(start, periods=days):
            for category, value in (("GC", 1.0), ("CL", 0.25), ("GG", 0.5)):
                if category == "CL" and not cl:
                    continue
                rows.append([customer, 2.0, 2259, category, date, *([value] * 48), ""])
    return pd.DataFrame(
        rows, columns=IDENTITY_COLUMNS + INTERVAL_COLUMNS + ["Row Quality"]
    )


def audit(frame):
    return audit_households({"train": frame}, {"train": ("2010-07-01", "2010-07-03")})[
        0
    ]


def test_literal_na_flag_remains_distinct_from_blank():
    frame = meter_frame()
    frame["date"] = frame["date"].dt.strftime("%d/%m/%Y")
    frame.loc[0, "Row Quality"] = "NA"
    text = io.StringIO("Ausgrid notice\n" + frame.to_csv(index=False))
    parsed = read_meter_csv(text)
    assert parsed.loc[0, "Row Quality"] == "NA"
    assert parsed.loc[1, "Row Quality"] == ""
    assert (
        "non_actual_or_unknown_quality_rows:1"
        in audit(parsed)[1]["exclusions"]["train"]
    )


def test_original_first_year_without_quality_column_is_accepted():
    frame = meter_frame().drop(columns="Row Quality")
    frame["date"] = frame["date"].dt.strftime("%d-%b-%y")
    parsed = read_meter_csv(io.StringIO("notice\n" + frame.to_csv(index=False)))
    assert parsed["Row Quality"].eq("").all()
    assert audit(parsed)[1]["eligible"]


def test_energy_conversion_gc_excludes_cl_and_pv_is_separate():
    result = meter_arrays(meter_frame(), [1], "2010-07-01", "2010-07-03")
    np.testing.assert_array_equal(result["loads_kw"], np.full((96, 1), 2.5))
    np.testing.assert_array_equal(result["pv_kw"], np.ones((96, 1)))
    assert result["loads_kw"].sum() * 0.5 == 120
    assert result["der_kw"].sum() * 0.5 == 48
    assert result["timestamps"][0] == "2010-07-01T00:00:00+10:00"
    assert result["timestamps"][47] == "2010-07-01T23:30:00+10:00"
    assert result["timestamps"][-1] == "2010-07-02T23:30:00+10:00"


def test_distinct_slots_days_and_reordered_households_remain_aligned():
    frame = meter_frame(customers=(1, 2))
    start = pd.Timestamp("2010-07-01")
    category_offsets = {"GC": 0, "CL": 10, "GG": 20}
    for index, row in frame.iterrows():
        base = row["Customer"] * 1000 + (row["date"] - start).days * 100
        frame.loc[index, INTERVAL_COLUMNS] = (
            base + category_offsets[row["Consumption Category"]] + np.arange(48)
        )
    result = meter_arrays(
        frame.sample(frac=1, random_state=5), [2, 1], "2010-07-01", "2010-07-03"
    )
    for t in range(96):
        day, slot = divmod(t, 48)
        for column, customer in enumerate((2, 1)):
            base = customer * 1000 + day * 100 + slot
            assert result["loads_kw"][t, column] == (2 * base + 10) / 0.5
            assert result["pv_kw"][t, column] == (base + 20) / 0.5


def test_absent_cl_meter_is_zero_but_missing_day_is_excluded():
    absent = meter_frame(cl=False)
    assert audit(absent)[1]["eligible"]
    np.testing.assert_array_equal(
        meter_arrays(absent, [1], "2010-07-01", "2010-07-03")["loads_kw"], 2.0
    )
    partial = meter_frame().drop(index=1)
    assert audit(partial)[1]["exclusions"]["train"] == ["CL_missing_days:1"]


def test_cl_absence_in_one_year_is_not_mistaken_for_no_meter():
    periods = {
        "train": ("2010-07-01", "2010-07-03"),
        "test": ("2011-07-01", "2011-07-03"),
    }
    audits, _ = audit_households(
        {"train": meter_frame(), "test": meter_frame(start="2011-07-01", cl=False)},
        periods,
    )
    assert audits[1]["controlled_load_meter"]
    assert audits[1]["exclusions"]["test"] == ["CL_missing_days:2"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1])
def test_nonfinite_and_negative_measurements_exclude_household(bad):
    frame = meter_frame()
    frame.loc[0, "0:30"] = bad
    assert audit(frame)[1]["exclusions"]["train"] == ["invalid_meter_values:1"]
    _, summary = audit_households(
        {"train": frame}, {"train": ("2010-07-01", "2010-07-03")}
    )
    if not np.isfinite(bad):
        assert summary["train"]["source_energy_kwh"]["GC"] is None


def test_duplicate_and_missing_gc_day_are_not_silently_repaired():
    frame = meter_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    assert "duplicate_channel_date" in audit(duplicate)[1]["exclusions"]["train"]
    missing = frame.drop(index=0)
    assert "GC_missing_days:1" in audit(missing)[1]["exclusions"]["train"]
    with pytest.raises(ValueError, match="Incomplete"):
        meter_arrays(missing, [1], "2010-07-01", "2010-07-03")


def test_postcode_selection_does_not_use_consumption_magnitude():
    frame = meter_frame(customers=(5, 4, 3, 2, 1))
    frame.loc[frame.Customer.isin([1, 2]), "Postcode"] = 1000
    original = select_households(audit(frame), 4)
    assert original == [3, 4, 5, 1]
    frame[INTERVAL_COLUMNS] *= 100
    assert select_households(audit(frame), 4) == original
    with pytest.raises(ValueError, match="only 5 eligible"):
        select_households(audit(frame), 6)


def test_nominal_calendar_retains_dst_day_48_slots_without_interpolation():
    frame = meter_frame(start="2010-10-03", days=1)
    result = meter_arrays(frame, [1], "2010-10-03", "2010-10-04")
    assert len(result["timestamps"]) == 48
    assert result["timestamps"][4] == "2010-10-03T02:00:00+10:00"
    assert TIME_POLICY["name"] == "nominal_calendar_not_reconstructed_UTC"


def test_leap_day_and_multi_household_aggregation():
    frame = meter_frame(start="2012-02-28", days=3, customers=(1, 2))
    result = meter_arrays(frame, [2, 1], "2012-02-28", "2012-03-02")
    assert result["loads_kw"].shape == (144, 2)
    assert result["timestamps"][48] == "2012-02-29T00:00:00+10:00"
    np.testing.assert_array_equal(result["der_kw"], result["pv_kw"].sum(axis=1))


def test_full_preparation_is_repeatable_and_hashes_bind_sources(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    periods = {
        "train": ("2010-2011", "2010-07-01", "2010-07-03"),
        "validation": ("2011-2012", "2011-07-01", "2011-07-03"),
        "test": ("2012-2013", "2012-07-01", "2012-07-03"),
    }
    monkeypatch.setattr(ausgrid, "SPLITS", periods)
    csvs = {}
    for year, start, _ in periods.values():
        frame = meter_frame(start=start)
        frame["date"] = frame["date"].dt.strftime("%d/%m/%Y")
        csvs[year] = ("notice\n" + frame.to_csv(index=False)).encode()
    with zipfile.ZipFile(raw / "Ausgrid_solar_home_data.zip", "w") as archive:
        archive.writestr(
            "Ausgrid solar home electricity data notes (Aug 2014).pdf", b"notes"
        )
        for year, data in csvs.items():
            archive.writestr(f"Solar home {year}.csv", data)
    with zipfile.ZipFile(raw / "independent_Data.zip", "w") as archive:
        archive.writestr(
            "RawData/2012-2013 Solar home electricity data v2.csv", csvs["2012-2013"]
        )
    (raw / "Ausgrid_notes_2014.pdf").write_bytes(b"notes")
    (raw / "origin_README.md").write_text("source", encoding="utf-8")
    (raw / "data_gov_au_metadata.json").write_text(
        json.dumps(
            {
                "result": {
                    "author": "Ausgrid",
                    "license_title": "CC BY 3.0 Australia",
                    "license_url": "https://creativecommons.org/licenses/by/3.0/au/",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ausgrid, "ARCHIVE_SHA256", ausgrid.sha256(raw / "Ausgrid_solar_home_data.zip")
    )
    path = ausgrid.prepare_dataset(tmp_path, households=1, download=False)
    first = {p.name: ausgrid.sha256(p) for p in path.parent.iterdir()}
    ausgrid.prepare_dataset(tmp_path, households=1, download=False)
    assert first == {p.name: ausgrid.sha256(p) for p in path.parent.iterdir()}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["units"]["loads_kw"] == "kW"
    assert manifest["time_policy"]["name"] == "nominal_calendar_not_reconstructed_UTC"
    assert manifest["preprocessing"]["code_sha256"] == ausgrid.sha256(
        ausgrid.Path(ausgrid.__file__)
    )
    assert manifest["splits"]["train"]["end_datetime"] == "2010-07-03T00:00:00+10:00"
    assert manifest["provenance"]["test_csv_independent_mirror_match"]
    with (raw / "Ausgrid_solar_home_data.zip").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive hash"):
        ausgrid.prepare_dataset(tmp_path, households=1, download=False)
