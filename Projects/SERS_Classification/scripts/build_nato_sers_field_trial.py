#!/usr/bin/env python3
"""Build an auditable manifest for the March 2024 NATO SERS field trial.

The source archive mixes Raman and SERS measurements, several vendor export
formats, duplicate conversions, instrument reports, and transcription errors.
This script does not modify the source archive.  It creates:

* a manifest of every recording-log row (expanded when a row names two scans),
* conservative SERS-only and QC-pass manifests,
* raw spectra interpolated onto a common 400--1800 cm^-1 grid, and
* a JSON audit summary.

The readable CSV/TXT exports are used for intensities.  Binary SPC and report
paths remain in the manifest for provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    value = 0
    for letter in letters.group(0):
        value = value * 26 + ord(letter) - 64
    return value - 1


def read_xlsx_rows(path: Path) -> dict[str, list[dict[int, Any]]]:
    """Read cell values without requiring openpyxl.

    The two source workbooks use simple cached values and do not require style
    or formula evaluation.  Sparse cell coordinates are preserved correctly.
    """

    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                for item in root
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }

        sheets: dict[str, list[dict[int, Any]]] = {}
        sheet_nodes = workbook.find(f"{{{MAIN_NS}}}sheets")
        if sheet_nodes is None:
            return sheets

        for sheet in sheet_nodes:
            name = sheet.attrib["name"]
            target = targets[sheet.attrib[f"{{{REL_NS}}}id"]]
            if not target.startswith("xl/"):
                target = f"xl/{target.lstrip('/')}"
            worksheet = ET.fromstring(archive.read(target))
            sheet_data = worksheet.find(f"{{{MAIN_NS}}}sheetData")
            rows: list[dict[int, Any]] = []
            if sheet_data is None:
                sheets[name] = rows
                continue

            for row in sheet_data:
                values: dict[int, Any] = {}
                for cell in row:
                    index = _column_index(cell.attrib["r"])
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    inline_node = cell.find(f"{{{MAIN_NS}}}is")
                    cell_type = cell.attrib.get("t")
                    if inline_node is not None:
                        value: Any = "".join(
                            node.text or ""
                            for node in inline_node.iter(f"{{{MAIN_NS}}}t")
                        )
                    elif value_node is None:
                        value = None
                    elif cell_type == "s":
                        value = shared_strings[int(value_node.text)]
                    elif cell_type == "b":
                        value = value_node.text == "1"
                    else:
                        value = value_node.text
                    if isinstance(value, str):
                        value = value.strip()
                    values[index] = value
                rows.append(values)
            sheets[name] = rows
        return sheets


def tabular_rows(
    sheet_rows: list[dict[int, Any]], header_row_index: int = 0
) -> Iterable[tuple[int, dict[str, Any]]]:
    if not sheet_rows:
        return
    header = {
        index: value.strip() if isinstance(value, str) else value
        for index, value in sheet_rows[header_row_index].items()
    }
    for excel_row, sparse_row in enumerate(
        sheet_rows[header_row_index + 1 :], start=header_row_index + 2
    ):
        row = {
            header[index]: value
            for index, value in sparse_row.items()
            if index in header and header[index]
        }
        if any(value not in (None, "") for value in row.values()):
            yield excel_row, row


def normalize_station(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "pills":
        return "pills"
    if text.startswith("surface"):
        return "surfaces"
    if text == "cwa":
        return "cwa"
    return None


def normalize_instrument(value: Any) -> str | None:
    text = str(value or "").lower().replace(" ", "").replace("-", "")
    for base, canonical in (
        ("agilent", "Agilent"),
        ("mira", "Mira"),
        ("pendar", "Pendar"),
        ("rmx", "RMX"),
    ):
        match = re.search(rf"{base}(\d)", text)
        if match:
            return f"{canonical}-{match.group(1)}"
    if text == "m1":
        return "Mira-1"
    if text == "pmcds":
        return "PMCDS"
    return None


def normalize_sensor(value: Any) -> tuple[str | None, str | None]:
    raw = str(value or "").strip()
    text = raw.lower()
    if not text or text == "na":
        return None, None

    psers = {
        "p-sers",
        "ag p-sers",
        "p-sers ag",
        "metrohm silver",
        "mr ag srs",
        "mag",
        "silver sers",
    }
    hkit = {"h-kit", "h-ag sers", "h-sers ag"}
    engineered = {"aggan", "augan", "agpol", "aupol"}
    if text in psers:
        family = "pSERS_Metrohm_silver"
    elif any(token in text for token in ("nrc", "nrs", "can")):
        family = "NRC_Canadian_SERS"
    elif text in hkit:
        family = "H_SERS_H_Kit"
    elif text in engineered:
        family = "GaN_polymer"
    else:
        family = "unmapped"

    variant_aliases = {
        "p-sers": "pSERS",
        "ag p-sers": "pSERS",
        "p-sers ag": "pSERS",
        "metrohm silver": "pSERS",
        "mr ag srs": "pSERS",
        "mag": "pSERS",
        "silver sers": "pSERS",
        "h-kit": "HKit",
        "h-ag sers": "HKit",
        "h-sers ag": "HKit",
        "can": "NRC_unspecified",
        "can-sers": "NRC_unspecified",
        "can-sers (ki)": "NRC_KI",
        "can sers": "NRC_unspecified",
        "canadian sers": "NRC_unspecified",
        "nrc sers": "NRC_unspecified",
        "nrc-sers canada": "NRC_unspecified",
        "nrc-srs": "NRC_unspecified",
        "nrc srs": "NRC_unspecified",
        "nrc-srs canada": "NRC_unspecified",
        "nrc sers can h2": "NRC_H2",
        "nrs sers can h2": "NRC_H2",
        "can-h2": "NRC_H2",
        "nrc sensor anh2": "NRC_ANH2",
        "nrc sers anh2": "NRC_ANH2",
        "can au sers": "NRC_Au",
        "aggan": "AgGaN",
        "augan": "AuGaN",
        "agpol": "AgPol",
        "aupol": "AuPol",
    }
    return family, variant_aliases.get(text, raw)


def normalize_target(description: str) -> str:
    text = description.lower()
    if "benzyl" in text or text.endswith("%bf"):
        return "benzyl_fentanyl"
    if "4anpp" in text:
        return "4_ANPP"
    if "aceta" in text:
        return "acetaminophen"
    if "blank" in text:
        return "blank"
    if "ethylparaxon" in text:
        return "ethyl_paraoxon"
    if "nitro" in text:
        return "4_nitrophenol"
    if "ethanol" in text:
        return "ethanol"
    return "unmapped"


def master_sample_attributes(station: str, description: str) -> dict[str, Any]:
    text = description.lower()
    if station == "pills":
        matrix = "pill"
        carrier_geometry = None
    elif station == "surfaces":
        matrix = "field_surface_coupon"
        if text.startswith("hexa-l"):
            carrier_geometry = "Hexa_L"
        elif text.startswith("sq-p"):
            carrier_geometry = "Square_P"
        else:
            carrier_geometry = None
    else:
        matrix = "cwa_solution"
        carrier_geometry = None

    concentration_match = re.search(r"(\d+)\s*mM", description, re.IGNORECASE)
    if concentration_match:
        nominal_concentration = f"{concentration_match.group(1)} mM"
    elif "5%" in description:
        nominal_concentration = "5%"
    else:
        nominal_concentration = None
    return {
        "sample_matrix": matrix,
        "carrier_geometry": carrier_geometry,
        "nominal_concentration": nominal_concentration,
    }


def master_sample_id(station: str | None, raw_sample: Any) -> int | None:
    text = str(raw_sample or "").strip()
    if station == "pills":
        match = re.match(r"^(\d+)", text)
        value = int(match.group(1)) if match else None
        return value if value is not None and 1 <= value <= 24 else None
    if station == "surfaces":
        match = re.search(r"(\d+)", text)
        value = int(match.group(1)) if match else None
        return value if value is not None and 25 <= value <= 49 else None
    if station == "cwa":
        if re.fullmatch(r"\d+", text):
            value = int(text)
            return value if 50 <= value <= 73 else None
        # Both C150 and C1S0 denote master sample 50.  This compact notation
        # is only used for samples 50--65 in the recording workbook.
        compact = re.match(r"^C\dS?(\d{2})", text, re.IGNORECASE)
        if compact:
            value = int(compact.group(1))
            return value if 50 <= value <= 73 else None
        split = re.match(r"^C([1-6])S(\d)", text, re.IGNORECASE)
        if split:
            scenario = int(split.group(1))
            suffix = int(split.group(2))
            scenario_start = 50 + 4 * (scenario - 1)
            candidates = range(scenario_start, scenario_start + 4)
            for candidate in candidates:
                if candidate % 10 == suffix:
                    return candidate
    return None


def parse_master_samples(path: Path) -> dict[int, dict[str, Any]]:
    sheets = read_xlsx_rows(path)
    result: dict[int, dict[str, Any]] = {}
    sheet_station = {"PILLS": "pills", "Surfaces": "surfaces", "CWA": "cwa"}
    for sheet_name, station in sheet_station.items():
        for _, row in tabular_rows(sheets[sheet_name]):
            sample_text = str(row.get("Sample #") or "")
            if not sample_text.isdigit():
                continue
            sample_id = int(sample_text)
            description = str(row.get("Description") or "").strip()
            result[sample_id] = {
                "master_sample_id": sample_id,
                "scenario": row.get("Scenario"),
                "master_station": station,
                "master_description": description,
                "target_analyte": normalize_target(description),
            }
            result[sample_id].update(master_sample_attributes(station, description))
    return result


def parse_recordings(path: Path, master: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    sheets = read_xlsx_rows(path)
    records: list[dict[str, Any]] = []
    for sheet_name, rows in sheets.items():
        if not sheet_name.startswith("Raman recordings"):
            continue
        session = int(sheet_name.rsplit(" ", 1)[-1])
        for excel_row, row in tabular_rows(rows):
            station = normalize_station(row.get("Station"))
            sample_id = master_sample_id(station, row.get("Sample #"))
            sensor_family, sensor_variant = normalize_sensor(
                row.get("Type of SERS substrate")
            )
            normal_flag = str(row.get("Normal Raman (Y/N)") or "").upper() or None
            named_sers = sensor_family is not None
            ambiguous_sers = normal_flag == "N" and not named_sers
            item: dict[str, Any] = {
                "recording_uid": f"session{session}_row{excel_row}",
                "session": session,
                "excel_row": excel_row,
                "sample_raw": row.get("Sample #"),
                "master_sample_id": sample_id,
                "team": str(row.get("Team (A, B, C)") or "").upper() or None,
                "operator_initials": row.get("Operator Initials"),
                "recorded_time": row.get("Time (24hrs)"),
                "station_raw": row.get("Station"),
                "station": station,
                "instrument_raw": row.get("Raman System"),
                "instrument": normalize_instrument(row.get("Raman System")),
                "normal_raman_flag": normal_flag,
                "sensor_raw": row.get("Type of SERS substrate"),
                "sensor_family": sensor_family,
                "sensor_variant": sensor_variant,
                "is_named_sers": named_sers,
                "is_ambiguous_sers": ambiguous_sers,
                "sensor_flag_conflict": bool(named_sers and normal_flag == "Y"),
                "instrument_result": row.get("Results"),
                "logged_file_name": row.get("File Name (number)"),
                "comments": row.get("Comments"),
                "target_detected_log": row.get("Target Det (Y/N)"),
                "paper_sheet": row.get("Sheet") or row.get("Sheet Recorded"),
            }
            if sample_id in master:
                item.update(master[sample_id])
            else:
                item.update(
                    {
                        "scenario": None,
                        "master_station": None,
                        "master_description": None,
                        "target_analyte": None,
                        "sample_matrix": None,
                        "carrier_geometry": None,
                        "nominal_concentration": None,
                    }
                )
            records.append(item)
    return records


def source_scan_ids(instrument: str | None, logged_name: Any) -> list[int]:
    text = str(logged_name or "").strip()
    if not text:
        return []
    if instrument == "Mira-1" and re.fullmatch(r"\d{4}-\d{4}", text):
        return [int(value) for value in text.split("-")]
    values = re.findall(r"\d+", text)
    return [int(values[-1])] if values else []


def _scan_index(directory: Path, glob: str, pattern: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob(glob):
        match = re.search(pattern, path.name, re.IGNORECASE)
        if match:
            result[int(match.group(1))] = path
    return result


def index_sources(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}

    for instrument, child in (("Agilent-1", "Agilent 1"), ("Agilent-3", "Agilent 3")):
        directory = root / "Agilent" / child
        spc = _scan_index(directory, "*.spc", r"Scan\s+(\d+)")
        aggregate = next(directory.glob("*.csv"))
        result[instrument] = {
            scan_id: {
                "source_text_path": str(aggregate),
                "source_spc_path": str(path),
                "source_pdf_path": None,
                "source_prb_path": None,
                "source_format": "Agilent aggregate CSV: Xaxis,SORS,bZero,bOffset",
            }
            for scan_id, path in spc.items()
        }

    for instrument, child in (("Mira-1", "Mira 1"), ("Mira-2", "Mira 2"), ("Mira-3", "Mira 3")):
        directory = root / "Mira" / child
        csv_index = _scan_index(directory, "*.csv", r"Scan\s+(\d+)")
        spc_index = _scan_index(directory, "*.spc", r"Scan\s+(\d+)")
        pdf_index = _scan_index(directory, "*.pdf", r"Scan\s+(\d+)")
        result[instrument] = {
            scan_id: {
                "source_text_path": str(path),
                "source_spc_path": str(spc_index.get(scan_id) or "") or None,
                "source_pdf_path": str(pdf_index.get(scan_id) or "") or None,
                "source_prb_path": None,
                "source_format": "Mira two-column CSV",
            }
            for scan_id, path in csv_index.items()
        }

    for instrument, child in (
        ("Pendar-1", "Pendar 1"),
        ("Pendar-2", "Pendar 2"),
        ("Pendar-3", "Pendar 3"),
    ):
        directory = root / "Pendar" / child
        csv_index = _scan_index(directory, "*.csv", r"^[^-]+-(\d+)-")
        result[instrument] = {
            scan_id: {
                "source_text_path": str(path),
                "source_spc_path": str(path.with_suffix(".spc")),
                "source_pdf_path": str(path.with_suffix(".pdf")),
                "source_prb_path": str(path.with_suffix(".prb")),
                "source_format": "Pendar two-column CSV",
            }
            for scan_id, path in csv_index.items()
        }

    for instrument, child in (("RMX-1", "RMX 1"), ("RMX-2", "RMX 2")):
        directory = root / "RMX" / child
        txt_index = _scan_index(directory, "Scan*.txt", r"Scan(\d+)")
        result[instrument] = {
            scan_id: {
                "source_text_path": str(path),
                "source_spc_path": str(path.with_suffix(".spc")),
                "source_dark_spc_path": str(path.with_name(f"{path.stem}-dark.spc")),
                "source_pdf_path": None,
                "source_prb_path": None,
                "source_format": "RMX metadata TXT: processed spectrum + dark spectrum",
            }
            for scan_id, path in txt_index.items()
        }
    return result


def attach_sources(
    records: list[dict[str, Any]], sources: dict[str, dict[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    corrections = {("Agilent-1", 1798): 2798}
    for record in records:
        ids = source_scan_ids(record["instrument"], record["logged_file_name"])
        if not ids:
            ids = [None]
        for subindex, original_scan_id in enumerate(ids):
            item = dict(record)
            resolved_scan_id = corrections.get(
                (record["instrument"], original_scan_id), original_scan_id
            )
            item["recording_subindex"] = subindex
            item["observation_uid"] = (
                record["recording_uid"]
                if len(ids) == 1
                else f"{record['recording_uid']}_scan{resolved_scan_id}"
            )
            item["source_scan_id_logged"] = original_scan_id
            item["source_scan_id"] = resolved_scan_id
            item["source_id_corrected"] = resolved_scan_id != original_scan_id
            source = sources.get(record["instrument"] or "", {}).get(resolved_scan_id)
            if source:
                item.update(source)
                item["source_match_status"] = "matched"
            else:
                for field in (
                    "source_text_path",
                    "source_spc_path",
                    "source_dark_spc_path",
                    "source_pdf_path",
                    "source_prb_path",
                    "source_format",
                ):
                    item[field] = None
                if record["instrument"] not in sources:
                    item["source_match_status"] = "instrument_data_unavailable"
                elif original_scan_id is None:
                    item["source_match_status"] = "no_scan_file_logged"
                else:
                    item["source_match_status"] = "scan_not_found"
            expanded.append(item)
    return expanded


def load_spectrum(record: dict[str, Any], agilent_cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    instrument = record["instrument"]
    path = Path(record["source_text_path"])
    if instrument and instrument.startswith("Agilent"):
        key = str(path)
        if key not in agilent_cache:
            with path.open(newline="") as handle:
                first_row = next(csv.reader(handle))
            values = np.loadtxt(path, delimiter=",", skiprows=2)
            scans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for index in range(values.shape[1] // 4):
                match = re.search(r"\d+", first_row[4 * index + 3])
                if match:
                    scans[int(match.group(0))] = (
                        values[:, 4 * index],
                        values[:, 4 * index + 1],
                    )
            agilent_cache[key] = scans
        return agilent_cache[key][int(record["source_scan_id"])]

    if instrument and instrument.startswith("Mira"):
        values = np.loadtxt(path, delimiter=",")
        return values[:, 0], values[:, 1]
    if instrument and instrument.startswith("Pendar"):
        values = np.loadtxt(path, delimiter=",", skiprows=1)
        return values[:, 0], values[:, 1]
    if instrument and instrument.startswith("RMX"):
        lines = path.read_text(errors="replace").splitlines()
        start = next(index for index, line in enumerate(lines) if line.startswith("spectrum "))
        count = int(lines[start].split()[1])
        values = np.asarray(
            [[float(value) for value in line.split()] for line in lines[start + 1 : start + 1 + count]]
        )
        return values[:, 0], values[:, 1]
    raise ValueError(f"No spectrum reader for {instrument}")


def add_reference_and_numeric_qc(records: list[dict[str, Any]]) -> None:
    by_source: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["source_match_status"] == "matched":
            by_source[(record["instrument"], int(record["source_scan_id"]))].append(record)

    for group in by_source.values():
        signatures = {
            (
                item["is_named_sers"],
                item["master_sample_id"],
                item["target_analyte"],
                item["sensor_family"],
            )
            for item in group
        }
        conflict = len(signatures) > 1
        for index, item in enumerate(group):
            item["source_reference_count"] = len(group)
            item["source_reference_conflict"] = conflict
            item["source_primary_reference"] = index == 0

    agilent_cache: dict[str, Any] = {}
    for record in records:
        record.setdefault("source_reference_count", 0)
        record.setdefault("source_reference_conflict", False)
        record.setdefault("source_primary_reference", False)
        text = " ".join(
            str(record.get(field) or "")
            for field in ("instrument_result", "comments", "logged_file_name")
        ).lower()
        record["manual_severe_qc_flag"] = bool(
            re.search(
                r"poor data|poor quality|\bpdq\b|saturat|no spectrum|"
                r"instrument died|timeout|time out|no signal|scan cancel|"
                r"no file|too long time|measurement crushed",
                text,
            )
        )
        record["manual_low_signal_or_noise_flag"] = bool(
            re.search(r"low signal|poor signal|\bnoise|noises|weak peak", text)
        )
        record["is_background_or_unresolved_control"] = (
            record["master_sample_id"] is None
            and str(record.get("sample_raw") or "").strip().lower()
            in {"bg", "blank", "-", ""}
        )
        record["numeric_qc_status"] = None
        if record["source_match_status"] != "matched":
            continue
        try:
            x, y = load_spectrum(record, agilent_cache)
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.all():
                status = "non_finite"
            elif len(x) < 2 or not np.all(np.diff(x) > 0):
                status = "invalid_axis"
            elif np.ptp(y) == 0:
                status = "constant"
            else:
                status = "pass"
            record.update(
                {
                    "numeric_qc_status": status,
                    "n_points": len(x),
                    "axis_min_cm1": float(x[0]),
                    "axis_max_cm1": float(x[-1]),
                    "intensity_min": float(np.min(y)),
                    "intensity_median": float(np.median(y)),
                    "intensity_max": float(np.max(y)),
                    "intensity_std": float(np.std(y)),
                    "negative_fraction": float(np.mean(y < 0)),
                }
            )
            baseline = gaussian_filter1d(
                percentile_filter(y, percentile=10, size=101, mode="nearest"), 20
            )
            record["baseline_energy_fraction_proxy"] = float(
                np.linalg.norm(baseline) / max(np.linalg.norm(y), 1e-12)
            )
            record["baseline_span_fraction_proxy"] = float(
                np.ptp(baseline) / max(np.ptp(y), 1e-12)
            )
        except Exception as error:  # preserve the failure in the audit manifest
            record["numeric_qc_status"] = f"load_error:{type(error).__name__}"

    for record in records:
        record["include_sers_core"] = bool(
            record["is_named_sers"]
            and record["source_match_status"] == "matched"
            and record["master_sample_id"] is not None
            and record["target_analyte"] not in (None, "unmapped")
            and not record["source_reference_conflict"]
            and record["source_primary_reference"]
            and record["numeric_qc_status"] == "pass"
        )
        record["include_sers_qc_pass"] = bool(
            record["include_sers_core"]
            and not record["manual_severe_qc_flag"]
            and not record["manual_low_signal_or_noise_flag"]
        )


def parse_report_metadata(record: dict[str, Any]) -> dict[str, Any]:
    path_text = record.get("source_pdf_path")
    if not path_text:
        return {}
    process = subprocess.run(
        ["pdftotext", "-layout", path_text, "-"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    text = process.stdout
    patterns = {
        "instrument_serial": r"Serial Number:\s*([^\n]+)|Serial Number\s+([^\n]+)",
        "firmware_package": r"Firmware Package\s+([^\n]+)",
        "software_version": r"Software Version:\s*([^\n]+)",
        "integration_time": r"Integration Time\s+([^\n]+)",
        "auto_integration": r"Auto Integration\s+([^\n]+)",
        "averages": r"Averages\s+([^\n]+)",
        "laser_power": r"Laser Power\s+([^\n]+)",
        "smart_tip_type": r"Smart Tip Type\s+([^\n]+)",
        "system_suitability": r"Last System Suitability Test\s+([^\n]+)",
        "measurement_duration": r"Duration:\s*([^\n]+)",
        "instrument_start_date": r"Start Date:\s*([^\n]+)",
        "instrument_start_time": r"Start Time:\s*([^\n]+)",
    }
    metadata: dict[str, Any] = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = next(value for value in match.groups() if value is not None).strip()
        metadata[field] = re.split(r"\s{3,}", value)[0].strip()
    return metadata


def parse_rmx_metadata(record: dict[str, Any]) -> dict[str, Any]:
    if not str(record.get("instrument") or "").startswith("RMX"):
        return {}
    path_text = record.get("source_text_path")
    if not path_text:
        return {}
    wanted = {
        "timestamp",
        "ccdbias",
        "ccdgain",
        "sigmaread",
        "laserwavenum",
        "wavenumcorrection",
        "scancount",
        "totalexposurems",
        "singleexposurems",
        "measurementstate",
        "scanmode",
        "replacelist",
        "peaks",
    }
    metadata: dict[str, Any] = {}
    for line in Path(path_text).read_text(errors="replace").splitlines():
        key, _, value = line.partition(" ")
        if key in wanted:
            metadata[f"rmx_{key}"] = value.strip()
    return metadata


def add_acquisition_metadata(records: list[dict[str, Any]]) -> None:
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if record["source_match_status"] != "matched":
            continue
        key = (record["instrument"], int(record["source_scan_id"]))
        if key not in cache:
            metadata = parse_report_metadata(record)
            metadata.update(parse_rmx_metadata(record))
            if record["instrument"] and record["instrument"].startswith("Agilent"):
                match = re.search(r"Resolve\s+(RES\d+)", str(record["source_spc_path"]))
                if match:
                    metadata["instrument_serial"] = match.group(1)
            cache[key] = metadata
        record.update(cache[key])


def write_common_grid_arrays(
    records: list[dict[str, Any]], output_path: Path, start: int = 400, end: int = 1800
) -> None:
    selected = [record for record in records if record["include_sers_core"]]
    axis = np.arange(start, end + 1, dtype=np.float32)
    intensities = np.empty((len(selected), len(axis)), dtype=np.float32)
    agilent_cache: dict[str, Any] = {}
    for index, record in enumerate(selected):
        x, y = load_spectrum(record, agilent_cache)
        intensities[index] = np.interp(axis, x, y).astype(np.float32)
    np.savez_compressed(
        output_path,
        axis_cm1=axis,
        intensity=intensities,
        observation_uid=np.asarray([record["observation_uid"] for record in selected]),
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def build_summary(records: list[dict[str, Any]], sources: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    named = frame[frame["is_named_sers"]]
    core = frame[frame["include_sers_core"]]
    qc = frame[frame["include_sers_qc_pass"]]
    return {
        "recording_observations": len(frame),
        "named_sers_recording_observations": len(named),
        "ambiguous_sers_recording_observations": int(frame["is_ambiguous_sers"].sum()),
        "normal_raman_or_non_sers_observations": int(
            (~frame["is_named_sers"] & ~frame["is_ambiguous_sers"]).sum()
        ),
        "source_spectra_indexed": {key: len(value) for key, value in sources.items()},
        "source_match_status_all": frame["source_match_status"].value_counts().to_dict(),
        "source_match_status_named_sers": named["source_match_status"].value_counts().to_dict(),
        "sensor_alias_counts": named["sensor_raw"].value_counts(dropna=False).to_dict(),
        "sensor_family_counts_named_sers": named["sensor_family"].value_counts().to_dict(),
        "sensor_family_counts_core": core["sensor_family"].value_counts().to_dict(),
        "instrument_counts_core": core["instrument"].value_counts().to_dict(),
        "target_counts_core": core["target_analyte"].value_counts().to_dict(),
        "core_observations": len(core),
        "qc_pass_observations": len(qc),
        "sensor_flag_conflicts": int(frame["sensor_flag_conflict"].sum()),
        "corrected_source_ids": int(frame["source_id_corrected"].sum()),
        "conflicting_source_references": int(frame["source_reference_conflict"].sum()),
        "manual_severe_qc_core": int(core["manual_severe_qc_flag"].sum()),
        "manual_low_signal_or_noise_core": int(
            core["manual_low_signal_or_noise_flag"].sum()
        ),
        "common_grid_cm1": [400, 1800, 1],
        "important_semantics": {
            "na": "No SERS sensor; normal Raman spectrum (exclude from SERS-only data).",
            "instrument_result": "Field instrument/library output, not ground truth.",
            "target_analyte": "Derived only from the master sample list.",
            "include_sers_core": "Named sensor, source matched, ground truth resolved, unambiguous source reference, valid numeric spectrum.",
            "include_sers_qc_pass": "Core plus conservative exclusion of severe/low-signal notes.",
        },
    }


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    default_source = repository.parents[1] / "2026July21" / "NATO SERS Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument(
        "--output-dir", type=Path, default=repository / "Workspace" / "nato_sers_field_trial"
    )
    parser.add_argument(
        "--skip-pdf-metadata",
        action="store_true",
        help="Skip extraction of Mira/Pendar report metadata via pdftotext.",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"NATO SERS source directory not found: {source_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    master = parse_master_samples(
        source_root / "MASTER SAMPLE LIST FOR NATO SERS CDT_MARCH 2024.xlsx"
    )
    records = parse_recordings(
        source_root / "01_Raman Team recordings_06Mar2024.xlsx", master
    )
    sources = index_sources(source_root)
    records = attach_sources(records, sources)
    add_reference_and_numeric_qc(records)
    if not args.skip_pdf_metadata:
        add_acquisition_metadata(records)

    frame = pd.DataFrame(records).sort_values(
        ["session", "excel_row", "recording_subindex"]
    )
    frame.to_csv(output_dir / "recordings_manifest.csv", index=False)
    frame[frame["include_sers_core"]].to_csv(
        output_dir / "sers_core_manifest.csv", index=False
    )
    frame[frame["include_sers_qc_pass"]].to_csv(
        output_dir / "sers_qc_pass_manifest.csv", index=False
    )
    write_common_grid_arrays(records, output_dir / "sers_core_spectra_raw_common_grid.npz")
    summary = build_summary(records, sources)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_ready) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_ready))


if __name__ == "__main__":
    main()
