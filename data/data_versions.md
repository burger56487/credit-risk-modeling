# Data versions, provenance and verification records

This file records which data was loaded, where it came from, what was verified
and what remains unproven. It contains no customer rows, no credentials and no
download tokens.

## Status (do not overstate it)

> 两文件选列数据库管道已通过人工样本的目标数据库集成验收，并完成真实规模首装、
> 数量与关联对账、同输入重复刷新，以及逐主键逐字段与表指纹核验。**官方来源证据
> 已补齐（与官方竞赛文件字节一致）**；旧库迁移与生产部署仍未覆盖。

## Input files (projection contract "数据契约第一版")

| File | Bytes | Data rows | Header fields | Loaded columns | Not loaded | SHA-256 |
|---|---|---|---|---|---|---|
| `application_train.csv` | 166,133,370 | 307,511 | 122 | 12 | 110 | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `bureau.csv` | 170,016,717 | 1,716,428 | 17 | 7 | 10 | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |

The two files live in `data/raw/` with their original names and were never
modified. They are git-ignored and are not part of this repository.

## Provenance

**Closed.** The operator downloaded the official competition archive
`home-credit-default-risk.zip` (721,616,255 bytes, 2026-09-15 12:29 local) from
`https://www.kaggle.com/competitions/home-credit-default-risk/data`, and the two
files it contains were compared by full SHA-256 against the digests of the input
files this project actually loaded:

| File | Official archive SHA-256 | Loaded input SHA-256 | Result |
|---|---|---|---|
| `application_train.csv` | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` | same | **byte-identical** |
| `bureau.csv` | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` | same | **byte-identical** |

Machine-readable evidence: `docs/final/official_provenance_check.json`. The
digests were computed by streaming the entries inside the official archive, so no
extraction step could alter them.

Consequence, per the agreed rule: the existing loads, the per-key content
verification and the frozen split all remain valid; **no fourth load and no
re-split were performed**. Byte identity establishes the content relationship only
— the official terms still govern use.

### Earlier provenance record (kept as history)

All four downloads came from **third-party Kaggle dataset pages**, not from the
official competition page. Evidence: the browser download records for the four
archives downloaded on 2026-09-14 between 17:06 and 17:08 local time. They are
**four third-party upload copies**: different uploaders may have copied the same
upstream file, so they cannot be treated as independent sources.

| Downloaded file | Page it was downloaded from | Bytes |
|---|---|---|
| `archive.zip` | `https://www.kaggle.com/datasets/andernoja/application-train-csv` | 37,847,529 |
| `archive (1).zip` | `https://www.kaggle.com/datasets/anggundwilestari/home-credit` | 735,679,967 |
| `archive (2).zip` (used) | `https://www.kaggle.com/datasets/julianocosta/home-credit` | 715,518,740 |
| `archive (3).zip` | `https://www.kaggle.com/datasets/youngdaniel/loan-dataset` | 721,558,302 |

What this supports:

- `application_train.csv` is byte-identical (full SHA-256) across all four
  third-party upload copies, including a single-file upload.
- `bureau.csv` is byte-identical (full SHA-256) across three of those copies.
- Both files' sizes match the entries recorded inside the archives used.

What this does **not** support:

- that either file is byte-identical to the official competition files;
- that the uploads were themselves faithful to the competition data.

Correction history: an earlier statement in this project's working notes described
`archive.zip` as an official single-file download and used it as an official
cross-check. **That claim is withdrawn (作废).** No official comparison exists in
this record yet.

### How the gap was closed (kept as history)

The official competition files require signing in to Kaggle and accepting the
competition rules, which neither this agent (no account, no credentials, no
browser control in this environment) nor the reviewing user could do on the
other's behalf. The operator downloaded the official archive from the Data tab
and provided it; the comparison rule used was **the SHA-256 of the files inside
that archive** (never the archive digest) against the recorded input digests.

| Outcome | Handling | Used here |
|---|---|---|
| Both files identical | Record the provenance link, keep the existing loads and verification evidence, freeze the same membership list; no fourth load | **this case** |
| Either file differs | Pause the freeze; check line endings, encoding or row order before field values or the record set | not applicable |
| Official files unavailable | Provenance stays blocked; no official real-data modelling | not applicable |

## Loads and code

| Item | Value |
|---|---|
| Contract version | 数据契约第一版 |
| Successful loads | `真实数据首装_第一版`, `真实数据首装_第二版`, `真实数据首装_第三版` |
| Failed loads | 0 |
| SQL script digests (prefix) | `00_staging_tables.sql` `526784c50636`, `01_create_tables.sql` `433fc7a328b8`, `02_aggregate_bureau.sql` `f655514c213a`, `03_build_model_table.sql` `b943ae5b456a` |
| Database | PostgreSQL 16.14, isolated local research instance (loopback only) |

## Content version of the four tables

Two different fingerprints were produced at different times. They are kept as
separate evidence, and neither is retroactively claimed for the earlier runs.

**(a) Aggregate checksum** used in the first working notes:
`sum(hashtextextended(row::text, 0))` per table. It is an order-independent
aggregate checksum, not a cryptographic digest, and it must not be quoted as proof
that two tables are identical.

**(b) Interim md5** (kept only as the evidence that the third load reproduced the
previous content): md5 over a header line plus one canonical line per row, ordered
by primary key, fields joined by `0x1f`, NULL as `\N`. Values: `application_train`
`10216b4c6cc0c216ff063d370f6c8308`, `bureau`
`4942a94e8ed446454bfad29007113cc5`, `feat_bureau`
`2f6999ced51c62b88a9df19dec502815`, `model_input`
`869a5d0834c694a1ec826a7400123262`. The same values were observed before and
after the third load of the same input.

**(c) Table fingerprint v1 (current registration fingerprint)** — SHA-256, the
same algorithm family as the source files, with an explicit and versioned
serialisation that cannot run two different contents into the same byte sequence:

- encoding UTF-8; rows in ascending primary-key order; fields in contract order;
- header: the literal `creditrisk-table-fingerprint v1`, the table name, then each
  column name length-prefixed with 4 big-endian bytes;
- each field: 1 tag byte (`N` for NULL, `V` for a value), then an 8-byte big-endian
  length, then the payload bytes;
- each row ends with `0x1e`.

This fingerprint describes **the current table version**; it was not computed on
the three earlier loads, and the earlier evidence is not replaced by it.

| Table | Rows | SHA-256 (fingerprint v1) |
|---|---|---|
| `application_train` | 307,511 | `4359bb186601a630dd4e16afb37abb9906a0ff9f096a45b47a95274170765be8` |
| `bureau` | 1,716,428 | `b8e47e86898404c7a22fec19f47bc75232f9823294ccb83ce5d07ce4fffc26a5` |
| `feat_bureau` | 305,811 | `d2f111307ce61a63fea2aefdc931a544ea8c5475a78a7a3a272e9722c0a23121` |
| `model_input` | 307,511 | `49480b96f6e55de5f2b41f245b62ed0153d781b5525e4f316e8df4ab013c8bd4` |

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

A separate, different state exists: for the 44,020 applications with **no bureau
record at all**, the count fields are zero while the amount fields and the
corresponding statistics are empty (`NULL`) by design. Do not merge that with
"record exists but the amount is unknown".

## Dependencies

After the lock file was regenerated against the official PyPI index (the earlier
lock referenced a mirror that GitHub runners receive 403 from), the package **name
and version set is unchanged** (73 packages). That supports "the dependency
version set did not change"; it does not by itself prove the built artefacts are
identical. Current `uv.lock` SHA-256:
`47b3b056b14122feb9e73229599c8db8708484bd97557570a2190b966106c2f6`.
Continuous integration for this revision: acceptance job (no database) 294 passed,
17 skipped, coverage with branch checking 80.11%; PostgreSQL job 311 passed, no
skips, coverage with branch checking 86.86%.

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
| Freeze status | **Frozen (2026-09-15)**, provenance closed: the input files are byte-identical to the official competition archive |

### Engineering freeze executed (provenance closed)

The membership list above was **used as the frozen split** for the real experiment
and the final evaluation, and is bound by digest inside
`artifacts/experiments/real_v1/freeze_manifest.json`: training 184,506 /
validation 61,502 / test 61,503, unit = application id, no re-sampling, no new
seed, file unchanged.

Both the split and the scheme are frozen, and the provenance item is closed (see
the provenance section above). The results in `docs/final_report.md` are labelled
accordingly.

| Item | Value |
|---|---|
| Validation inner split | calibration 30,751 / selection 30,751, stratified, seed 42, digest `b264424c88bd96ce12327a32f1c63f1f8510c499615df0a95cf2e6d1f2823f37` |
| Chosen scheme | main model (gradient boosting) with **raw** probabilities; calibration made no material difference (log-loss gap ≈ 1e-6) |
| Frozen threshold | `0.1753`, rule: flagged when predicted probability is greater than or equal to the threshold; demo capacity 10% |
| Test set openings | 1 (recorded in `artifacts/final/test_set_openings.json`) |
| Final evaluation | `docs/final_report.md`; aggregate tables in `docs/final/`; per-applicant predictions stay local |

The provenance check passed and this same list was frozen: **no re-sampling, no
new seed, no rewriting of the membership file**. The metadata file records the
freeze time, the associated data version, the provenance evidence and the code
version. Reporting the sample sizes and label ratios needed for a stratified split
is not the same as having evaluated a model on the final test split; later work
must not adjust the split, the variables or the thresholds based on test-set
metrics.

## Artefact separation

Real-data artefacts: the research database (`C:\Users\Administrator\pg\
creditrisk-research`), `artifacts/partitions/real_v1/`.

Synthetic demo artefacts (must not be confused with real ones, and are not used
by any real-data step): `data/processed/demo_model_table.csv`,
`reports/*_demo/`, `artifacts/releases/演示包*`, `artifacts/monitoring/runs_demo.sqlite3`.
