# Data versions, provenance and verification records

This file records which data was loaded, where it came from, what was verified
and what remains unproven. It contains no customer rows, no credentials and no
download tokens.

## Status (do not overstate it)

> 两文件选列数据库管道已通过人工样本的目标数据库集成验收，并完成真实规模首装、
> 数量与关联对账及同输入重复刷新。**来源追溯（官方来源证据）与部分内容核验口径
> 待补齐**；旧库迁移与真实模型独立评价尚未验收。

## Input files (projection contract "数据契约第一版")

| File | Bytes | Data rows | Header fields | Loaded columns | Not loaded | SHA-256 |
|---|---|---|---|---|---|---|
| `application_train.csv` | 166,133,370 | 307,511 | 122 | 12 | 110 | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `bureau.csv` | 170,016,717 | 1,716,428 | 17 | 7 | 10 | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |

The two files live in `data/raw/` with their original names and were never
modified. They are git-ignored and are not part of this repository.

## Provenance (incomplete on purpose)

All four downloads came from **third-party Kaggle dataset pages**, not from the
official competition page. Evidence: the browser download records for the four
archives downloaded on 2026-09-14 between 17:06 and 17:08 local time.

| Downloaded file | Page it was downloaded from | Bytes |
|---|---|---|
| `archive.zip` | `https://www.kaggle.com/datasets/andernoja/application-train-csv` | 37,847,529 |
| `archive (1).zip` | `https://www.kaggle.com/datasets/anggundwilestari/home-credit` | 735,679,967 |
| `archive (2).zip` (used) | `https://www.kaggle.com/datasets/julianocosta/home-credit` | 715,518,740 |
| `archive (3).zip` | `https://www.kaggle.com/datasets/youngdaniel/loan-dataset` | 721,558,302 |

What this supports:

- `application_train.csv` is byte-identical (full SHA-256) across all four
  independent uploads, including a single-file upload.
- `bureau.csv` is byte-identical (full SHA-256) across three independent uploads.
- Both files' sizes match the entries recorded inside the archives used.

What this does **not** support:

- that either file is byte-identical to the official competition files;
- that the uploads were themselves faithful to the competition data.

An earlier statement in this project's working notes described `archive.zip` as
an official single-file download. That was wrong and is corrected here. To close
the provenance gap, download `application_train.csv` and `bureau.csv` from the
official competition page (after accepting its rules) and record their digests in
the table above.

## Loads and code

| Item | Value |
|---|---|
| Contract version | 数据契约第一版 |
| Successful loads | `真实数据首装_第一版`, `真实数据首装_第二版`, `真实数据首装_第三版` |
| Failed loads | 0 |
| SQL script digests (prefix) | `00_staging_tables.sql` `526784c50636`, `01_create_tables.sql` `433fc7a328b8`, `02_aggregate_bureau.sql` `f655514c213a`, `03_build_model_table.sql` `b943ae5b456a` |
| Database | PostgreSQL 16.14, isolated local research instance (loopback only) |

## Content version of the four tables

Algorithm: md5 over a header line (column names joined by `|`) followed by one
canonical line per row, rows ordered by primary key; fields joined by `0x1f`,
NULL rendered as `\N`, numbers rendered by their exact text form.

| Table | Rows | md5 |
|---|---|---|
| `application_train` | 307,511 | `10216b4c6cc0c216ff063d370f6c8308` |
| `bureau` | 1,716,428 | `4942a94e8ed446454bfad29007113cc5` |
| `feat_bureau` | 305,811 | `2f6999ced51c62b88a9df19dec502815` |
| `model_input` | 307,511 | `869a5d0834c694a1ec826a7400123262` |

These digests were identical before and after the third load of the same input,
so the repeated refresh produced byte-identical table content. (An earlier
working note quoted `sum(hashtextextended(row::text, 0))`. That is an *aggregate
checksum*, not a cryptographic digest, and it is superseded by the values above.)

## Verification results

| Check | Result |
|---|---|
| Primary-key duplicates, all four tables | 0 / 0 / 0 / 0 |
| Wide-table keys vs application keys (set difference) | 0 |
| All 11 application fields unchanged across the join | 0 mismatches |
| Per-key, per-field comparison, `application_train` | 307,511 rows × 12 columns, 0 mismatches; 0 missing keys |
| Per-key, per-field comparison, `bureau` | 1,716,428 rows × 7 columns, 0 mismatches; 0 missing keys |
| Empty field in file vs NULL in database | 13/13, 257,669/257,669, 12/12, 173,378/173,378 (matching per column) |
| Special code `days_employed = 365243` | 55,374 in file = 55,374 in database |
| Column sums, non-integer value counts | identical per column |

Missing-value buckets, scoped to **applications that are in the wide table and
have at least one bureau record** (263,491 applications), counted **per
application**, per amount field, one field at a time:

| Aggregated field | All values unknown | Partially known | All values known |
|---|---|---|---|
| `bureau_credit_sum_total` (from `bureau.amt_credit_sum`) | 1 | 2 | 263,488 |
| `bureau_credit_sum_avg` (same known set) | 1 | 2 | 263,488 |
| `bureau_debt_total` (from `bureau.amt_credit_sum_debt`) | 7,360 | 102,086 | 154,045 |

A separate, different state exists: 44,020 applications have **no bureau record at
all**, so every bureau-derived value is NULL by design. Do not merge that with
"record exists but the amount is unknown".

## Observed performance (only what was measured)

| Observation | Value |
|---|---|
| First load wall time | 36.1 s |
| Second load (same input) wall time | 48.1 s |
| Third load (same input) | ran successfully; not separately timed |
| Input bytes | 336,150,087 |
| Database size | 8 MB → 327 MB → 598 MB |
| Cluster directory total (data + WAL) | 1,725,296,025 bytes |
| Stage timings, peak memory, temp space, WAL increment | **not measured** |

The repeated refresh was observed to take longer. Delete-and-insert has extra
write and old-row-version overhead, but the contribution of each factor was not
measured here, so a specific cause cannot be confirmed. Two observations are not
a performance baseline, and the directory total above is not a WAL-increment
measurement.

## Sample membership (draft, not frozen)

| Item | Value |
|---|---|
| Source | `model_input` after `真实数据首装_第三版` |
| Split parameters | `valid_size=0.2`, `test_size=0.2`, `random_state=42`, stratified by label, unit = application id |
| Sizes | train 184,506 / valid 61,502 / test 61,503 (307,511 total) |
| Bad rate per set | 0.0807 / 0.0807 / 0.0807 |
| Disjoint and complete | verified (pairwise disjoint, union equals all applications) |
| `membership.csv` SHA-256 | `1dbe7fb19ea7f8ad76ea19b63df0f61e093e77aed97b9422e683dee5a0bdeda8` |
| Location | `artifacts/partitions/real_v1/` (local only; contains application ids, not published) |
| Freeze status | **Not frozen** — waiting for official provenance or an explicit decision to accept third-party provenance |

## Artefact separation

Real-data artefacts: the research database (`C:\Users\Administrator\pg\
creditrisk-research`), `artifacts/partitions/real_v1/`.

Synthetic demo artefacts (must not be confused with real ones, and are not used
by any real-data step): `data/processed/demo_model_table.csv`,
`reports/*_demo/`, `artifacts/releases/演示包*`, `artifacts/monitoring/runs_demo.sqlite3`.
