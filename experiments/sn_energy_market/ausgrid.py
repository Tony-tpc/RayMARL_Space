"""Download and prepare the measured Ausgrid household/PV benchmark.

Run ``python -m experiments.sn_energy_market.ausgrid`` from the repository root.
Only structural data quality determines the fixed household cohort; calibration
and scenario prices belong to the experiment adapter, not this measurement file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip"
ARCHIVE_SHA256 = "5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82"
SOURCE_URLS = {
    "Ausgrid_solar_home_data.zip": ARCHIVE_URL,
    "data_gov_au_metadata.json": (
        "https://data.gov.au/data/api/3/action/package_show?id=nsw-solar-home-electricty-data"
    ),
    "origin_README.md": (
        "https://raw.githubusercontent.com/pierre-haessig/ausgrid-solar-data/master/README.md"
    ),
    "Ausgrid_notes_2014.pdf": (
        "https://raw.githubusercontent.com/rockey-abhishek/Ausgrid-Dataset/main/"
        "Ausgrid%20solar%20home%20electricity%20data%20notes%20%28Aug%202014%29.pdf"
    ),
    "independent_Data.zip": (
        "https://raw.githubusercontent.com/rockey-abhishek/Ausgrid-Dataset/main/Data.zip"
    ),
}
SPLITS = {
    "train": ("2010-2011", "2010-07-01", "2011-07-01"),
    "validation": ("2011-2012", "2011-07-01", "2012-07-01"),
    "test": ("2012-2013", "2012-07-01", "2013-07-01"),
}
INTERVAL_COLUMNS = [
    f"{minute // 60}:{minute % 60:02d}" for minute in range(30, 1440, 30)
] + ["0:00"]
IDENTITY_COLUMNS = [
    "Customer",
    "Generator Capacity",
    "Postcode",
    "Consumption Category",
    "date",
]
TIME_POLICY = {
    "name": "nominal_calendar_not_reconstructed_UTC",
    "fixed_offset": "+10:00",
    "source_labels": "interval end, 00:30 through following midnight 00:00",
    "output_labels": "interval start, 00:00 through 23:30",
    "interpretation": (
        "Preserve all 48 published slots per source date on a fixed-offset simulation clock. "
        "Original notes mention EST/EDT but do not disambiguate DST transition slots. "
        "No slot is inserted, dropped, interpolated or moved between dates; "
        "these timestamps do not claim reconstructed physical UTC alignment."
    ),
    "dst_transition_dates": [
        "2010-10-03",
        "2011-04-03",
        "2011-10-02",
        "2012-04-01",
        "2012-10-07",
        "2013-04-07",
    ],
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_sources(raw_dir: Path) -> dict:
    """Cache source documents and verify the pinned raw archive before processing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    access_path = raw_dir / "download_log.json"
    access_log = (
        json.loads(access_path.read_text(encoding="utf-8"))
        if access_path.exists()
        else []
    )
    for name, url in SOURCE_URLS.items():
        path = raw_dir / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            with (
                urlopen(
                    Request(
                        url,
                        headers={"User-Agent": "RayMARL research dataset preparation"},
                    ),
                    timeout=60,
                ) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(path)
            access_log.append(
                {
                    "file": name,
                    "url": url,
                    "downloaded_utc": datetime.now(UTC).isoformat(),
                }
            )
        sources[name] = {
            "url": url,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    access_path.write_text(json.dumps(access_log, indent=2) + "\n", encoding="utf-8")
    if sources["Ausgrid_solar_home_data.zip"]["sha256"] != ARCHIVE_SHA256:
        raise ValueError("Ausgrid archive hash differs from the verified release")
    return sources


def read_meter_csv(stream) -> pd.DataFrame:
    """Read literal quality flags; pandas' default NA parsing would erase 'NA'."""
    frame = pd.read_csv(stream, skiprows=1, keep_default_na=False)
    expected = IDENTITY_COLUMNS + INTERVAL_COLUMNS
    if list(frame.columns) not in (expected, expected + ["Row Quality"]):
        raise ValueError("Unexpected Ausgrid meter CSV header")
    sample = str(frame["date"].iloc[0])
    frame["date"] = pd.to_datetime(
        frame["date"], format="%d-%b-%y" if "-" in sample else "%d/%m/%Y"
    )
    if not set(frame["Consumption Category"]).issubset({"GC", "CL", "GG"}):
        raise ValueError("Unknown Ausgrid consumption category")
    for field in INTERVAL_COLUMNS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "Row Quality" not in frame:
        frame["Row Quality"] = ""
    return frame


def audit_households(
    frames: dict[str, pd.DataFrame], periods: dict
) -> tuple[dict, dict]:
    """Check every date/channel and preserve per-household exclusion reasons."""
    ids = sorted(set().union(*(set(frame["Customer"]) for frame in frames.values())))
    audits, summaries = {}, {}
    for split, frame in frames.items():
        values = frame[INTERVAL_COLUMNS].to_numpy(dtype=float)
        totals = {}
        for category in ("GC", "CL", "GG"):
            channel = frame.loc[
                frame["Consumption Category"] == category, INTERVAL_COLUMNS
            ].to_numpy(float)
            totals[category] = (
                float(channel.sum()) if np.isfinite(channel).all() else None
            )
        summaries[split] = {
            "rows": len(frame),
            "households": frame["Customer"].nunique(),
            "quality_flags": frame["Row Quality"].value_counts().to_dict(),
            "invalid_meter_values": int((~np.isfinite(values) | (values < 0)).sum()),
            "source_energy_kwh": totals,
        }
    for customer in ids:
        pieces = {
            split: frame.loc[frame["Customer"] == customer]
            for split, frame in frames.items()
        }
        all_rows = pd.concat(pieces.values())
        categories = set(all_rows["Consumption Category"])
        expected_categories = {"GC", "GG"} | ({"CL"} if "CL" in categories else set())
        record = {
            "postcode": int(all_rows["Postcode"].iloc[0]),
            "generator_capacity_kwp": float(all_rows["Generator Capacity"].iloc[0]),
            "controlled_load_meter": "CL" in categories,
            "exclusions": {},
        }
        if (
            all_rows["Postcode"].nunique() != 1
            or all_rows["Generator Capacity"].nunique() != 1
        ):
            record["exclusions"]["metadata"] = [
                "postcode_or_generator_capacity_changed"
            ]
        for split, part in pieces.items():
            start, end = periods[split]
            expected_dates = pd.date_range(start, end, inclusive="left", freq="D")
            reasons = []
            if part.duplicated(["Consumption Category", "date"]).any():
                reasons.append("duplicate_channel_date")
            if not part["date"].isin(expected_dates).all():
                reasons.append("date_outside_split")
            for category in sorted(expected_categories):
                dates = part.loc[part["Consumption Category"] == category, "date"]
                missing = len(expected_dates.difference(dates))
                if missing:
                    reasons.append(f"{category}_missing_days:{missing}")
            values = part[INTERVAL_COLUMNS].to_numpy(dtype=float)
            invalid = int((~np.isfinite(values) | (values < 0)).sum())
            if invalid:
                reasons.append(f"invalid_meter_values:{invalid}")
            flags = int(part["Row Quality"].ne("").sum())
            if flags:
                reasons.append(f"non_actual_or_unknown_quality_rows:{flags}")
            if reasons:
                record["exclusions"][split] = reasons
        record["eligible"] = not record["exclusions"]
        audits[int(customer)] = record
    return audits, summaries


def select_households(audits: dict, count: int) -> list[int]:
    """Prefer the postcode with most structurally complete actual-meter homes."""
    by_postcode = {}
    for customer, record in audits.items():
        if record["eligible"]:
            by_postcode.setdefault(record["postcode"], []).append(customer)
    ordering = sorted(by_postcode, key=lambda code: (-len(by_postcode[code]), code))
    selected = [
        customer for code in ordering for customer in sorted(by_postcode[code])
    ][:count]
    if count < 1 or len(selected) != count:
        raise ValueError(
            f"Requested {count} households; only {sum(map(len, by_postcode.values()))} eligible"
        )
    return selected


def meter_arrays(
    frame: pd.DataFrame, household_ids: list[int], start: str, end: str
) -> dict:
    """Preserve measured half-hour energy while changing units to average kW."""
    dates = pd.date_range(start, end, inclusive="left", freq="D")
    channels = {category: [] for category in ("GC", "CL", "GG")}
    for customer in household_ids:
        part = frame.loc[frame["Customer"] == customer]
        for category, channel_values in channels.items():
            rows = part.loc[part["Consumption Category"] == category].set_index("date")
            values = (
                np.zeros(len(dates) * 48)
                if category == "CL" and rows.empty
                else rows.reindex(dates)[INTERVAL_COLUMNS].to_numpy(float).reshape(-1)
            )
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(
                    f"Incomplete/invalid {category} for household {customer}"
                )
            channel_values.append(values / 0.5)
    gc, cl, pv = (np.stack(channels[key], axis=1) for key in ("GC", "CL", "GG"))
    dates = pd.date_range(f"{start}T00:00:00+10:00", periods=len(gc), freq="30min")
    return {
        "loads_kw": gc + cl,
        "pv_kw": pv,
        "der_kw": pv.sum(axis=1),
        "timestamps": np.array([value.isoformat() for value in dates]),
    }


def prepare_dataset(root: Path, *, households: int = 16, download: bool = True) -> Path:
    raw_dir, output = root / "raw", root / "processed"
    if download:
        sources = download_sources(raw_dir)
    else:
        sources = {
            name: {
                "url": url,
                "sha256": sha256(raw_dir / name),
                "bytes": (raw_dir / name).stat().st_size,
            }
            for name, url in SOURCE_URLS.items()
        }
        if sources["Ausgrid_solar_home_data.zip"]["sha256"] != ARCHIVE_SHA256:
            raise ValueError("Ausgrid archive hash differs from the verified release")
    frames, members = {}, {}
    with zipfile.ZipFile(raw_dir / "Ausgrid_solar_home_data.zip") as archive:
        for split, (year, _, _) in SPLITS.items():
            name = f"Solar home {year}.csv"
            with archive.open(name) as stream:
                frames[split] = read_meter_csv(stream)
            with archive.open(name) as stream:
                members[name] = {
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    "bytes": archive.getinfo(name).file_size,
                }
        notes = archive.read("Ausgrid solar home electricity data notes (Aug 2014).pdf")
        notes_sha = hashlib.sha256(notes).hexdigest()
        if notes_sha != sources["Ausgrid_notes_2014.pdf"]["sha256"]:
            raise ValueError("Original notes differ between independent mirrors")
        members["Ausgrid solar home electricity data notes (Aug 2014).pdf"] = {
            "sha256": notes_sha,
            "bytes": len(notes),
        }
    with (
        zipfile.ZipFile(raw_dir / "independent_Data.zip") as independent,
        independent.open(
            "RawData/2012-2013 Solar home electricity data v2.csv"
        ) as stream,
    ):
        independent_csv_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if independent_csv_sha != members["Solar home 2012-2013.csv"]["sha256"]:
        raise ValueError("2012–2013 CSV differs between independent mirrors")
    periods = {key: (start, end) for key, (_, start, end) in SPLITS.items()}
    audits, summaries = audit_households(frames, periods)
    household_ids = select_households(audits, households)
    output.mkdir(parents=True, exist_ok=True)
    split_records = {}
    for split, (_, start, end) in SPLITS.items():
        arrays = meter_arrays(frames[split], household_ids, start, end)
        path = output / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        selected = frames[split].loc[frames[split]["Customer"].isin(household_ids)]
        source_load = float(
            selected.loc[
                selected["Consumption Category"].isin(["GC", "CL"]), INTERVAL_COLUMNS
            ]
            .to_numpy(float)
            .sum()
        )
        source_pv = float(
            selected.loc[selected["Consumption Category"] == "GG", INTERVAL_COLUMNS]
            .to_numpy(float)
            .sum()
        )
        residuals = {
            "load_kwh": abs(float(arrays["loads_kw"].sum()) * 0.5 - source_load),
            "pv_kwh": abs(float(arrays["der_kw"].sum()) * 0.5 - source_pv),
        }
        if max(residuals.values()) > 1e-7:
            raise ValueError(f"Energy preservation failed for {split}: {residuals}")
        split_records[split] = {
            "file": path.name,
            "sha256": sha256(path),
            "periods": len(arrays["der_kw"]),
            "start_datetime": f"{start}T00:00:00+10:00",
            "end_datetime": f"{end}T00:00:00+10:00",
            "source_selected_load_kwh": source_load,
            "source_selected_pv_kwh": source_pv,
            "energy_residuals": residuals,
        }
    metadata = json.loads(
        (raw_dir / "data_gov_au_metadata.json").read_text(encoding="utf-8")
    )["result"]
    manifest = {
        "dataset": "ausgrid_solar_home",
        "title": "Ausgrid measured load and gross PV, fixed household cohort",
        "time_policy": TIME_POLICY,
        "units": {"loads_kw": "kW", "pv_kw": "kW", "der_kw": "kW"},
        "resolution_minutes": 30,
        "household_ids": household_ids,
        "consumers": [
            {"household_id": value, **audits[value]} for value in household_ids
        ],
        "splits": split_records,
        "preprocessing": {
            "code_sha256": sha256(Path(__file__)),
            "dependencies": {"numpy": np.__version__, "pandas": pd.__version__},
            "units": {
                "raw": "kWh per 30-minute interval",
                "loads_kw": "kW",
                "pv_kw": "kW",
                "der_kw": "kW",
            },
            "load_formula": "(GC + CL) / 0.5",
            "pv_formula": "GG / 0.5; DER is sum over selected households",
            "time_policy": TIME_POLICY,
            "selection": (
                "Structural completeness and actual-meter quality across all three years define availability eligibility. "
                "This cohort uses held-out availability metadata, but no held-out consumption magnitude, "
                "fitted statistic, price, reward or policy performance. Rank postcodes by eligible count descending, "
                "postcode ascending on ties, then household ID ascending; take the requested count."
            ),
            "absent_cl": "Zero only when the household has no CL rows in any of the three years; partial channels exclude the household.",
            "quality": "Blank quality only; literal NA and every other nonblank flag exclude the household.",
            "missing_values": "No imputation, interpolation, clipping or resampling.",
            "scenario_fields": "Flexibility, rigid load, prices and resource capacities are configured outside measured NPZ files.",
        },
        "quality_audit": {
            "total_households": len(audits),
            "eligible_households": sum(x["eligible"] for x in audits.values()),
            "by_split": summaries,
            "households": audits,
        },
        "provenance": {
            "sources": sources,
            "archive_members": members,
            "notes_independent_mirror_match": True,
            "test_csv_independent_mirror_match": True,
            "test_csv_independent_sha256": independent_csv_sha,
            "author": metadata["author"],
            "license_title": metadata["license_title"],
            "license_url": metadata["license_url"],
            "attribution": "Ausgrid, Solar Home Electricity Data, 2010–2013, released 2014; archive mirror Pierre Haessig, May 2026.",
            "transformations": "Selected complete actual-meter cohort, GC+CL aggregation, kWh-to-kW conversion, nominal interval-start calendar labels.",
        },
    }
    path = output / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/ausgrid"))
    parser.add_argument("--households", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    path = prepare_dataset(
        args.root, households=args.households, download=not args.no_download
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "manifest": str(path),
                "households": manifest["household_ids"],
                "eligible": manifest["quality_audit"]["eligible_households"],
                "periods": {
                    key: value["periods"] for key, value in manifest["splits"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
