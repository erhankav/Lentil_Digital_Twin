# Measurement archive

`lentil_microwave_archive.csv` contains the 979 records used in the study: free-space microwave transmission spectra
of two lentil cultivars measured over four harvest seasons, with the reference moisture content (MC) and the
gravimetric bulk density of every record.

## Material and measurement

- **Material:** Ceren (green) and Çiftçi (red) lentils supplied by the Central Research Institute for Field Crops,
  Ankara; harvest seasons 2014–2017 (S1–S4). Samples were conditioned to several MC levels per season.
- **Machine:** free-space transmission between two horn antennas, 2.00–4.00 GHz in 50 MHz steps (41 frequencies),
  15 acquisitions averaged per frequency; sample box 200 × 200 mm at the base and 240 mm high (fill volume 9600 cm³,
  200 mm along the propagation direction) on an integrated 20 kg load cell.
- **Procedure:** each filling started with the box filled to full height and levelled; in each of four further packing
  states the mass in the box was increased and the surface levelled again (five packing states per filling).
- **Reference MC:** oven method, 130 ± 1 °C for 19 h, three subsamples per MC level, % wet basis.
- **Spectra:** attenuation and phase differences relative to the empty-box spectrum stored at the start of each season;
  ε′, ε″ and tan δ follow from these differences through the free-space relations given in Section 2.2 of the paper.

Full details are given in Sections 2.1, 2.2 and 2.5 of the paper.

## File format

Comma-separated values, UTF-8, decimal point, one header row. One row per record and frequency: 979 records × 41
frequencies = 40,139 rows. A record is identified by `Cultivar`, `Year` and `Record_No`.

| Column | Unit | Description |
|---|---|---|
| `Cultivar` | – | `CEREN` (green) or `CIFTCI` (red) |
| `Year` | – | Harvest season: 2014 = S1, 2015 = S2, 2016 = S3, 2017 = S4 |
| `Level_Key` | – | MC level identifier: cultivar, year and level number, e.g. `CEREN_2015_L06` |
| `Level_ID` | – | MC level number within cultivar and season |
| `Filling_ID` | – | Filling number within cultivar and season |
| `Packing_State` | – | Packing state within the filling, 1–5 |
| `Record_No` | – | Record number in measurement order within cultivar and season |
| `Frequency_MHz` | MHz | Measurement frequency, 2000–4000 |
| `MC_wb_percent` | % w.b. | Reference MC stored for the record (wet basis) |
| `Bulk_Density_g_cm3` | g cm⁻³ | Gravimetric bulk density ρg = net sample mass / 9600 cm³ |
| `Phase_Shift_deg` | ° | Phase difference Δφ relative to the empty-box spectrum (unwrapped) |
| `Attenuation_dB` | dB | Attenuation difference ΔA relative to the empty-box spectrum (negative = loss) |
| `Eps_Real` | – | Dielectric constant ε′ |
| `Eps_Imag` | – | Dielectric loss factor ε″ |
| `Loss_Tangent` | – | Loss tangent tan δ = ε″/ε′ |

`MC_wb_percent` and `Bulk_Density_g_cm3` are properties of the record and repeat on its 41 rows.

## Records

| Season | Ceren: records (MC levels × fillings) | Çiftçi: records (MC levels × fillings) |
|---|---|---|
| S1 (2014) | 60 (12 × 1) | 80 (16 × 1) |
| S2 (2015) | 94 (19 × 1) | 85 (17 × 1) |
| S3 (2016) | 225 (15 × 3) | 225 (15 × 3) |
| S4 (2017) | 105 (7 × 3) | 105 (7 × 3) |
| **Total** | **484** | **495** |

Reference MC ranges from 9.99 to 19.74 % w.b. and ρg from 0.766 to 0.906 g cm⁻³.

## Notes

- **Structure columns.** The archive files store no level, filling or packing-state indices. `Level_ID`,
  `Filling_ID`, `Packing_State` and `Level_Key` were reconstructed from the record order: a new filling starts where
  ρg drops relative to the preceding record, the packing state is the position of the record within its filling, and
  fillings are assigned to MC levels in order (one per level in S1 and S2, three in S3 and S4). The code is
  `analysis/prepare_archive.py`; the raw files it reads (one Excel file per cultivar and season) are not part of the
  repository.
- **Excluded entries.** Six MC-level blocks without MC, mass or spectral values and one empty packing state were
  excluded before the analysis (Section 2.5 of the paper). Level numbers in this file are consecutive after that
  exclusion and can therefore differ from the laboratory numbering quoted in the paper. Level `CEREN_2015_L13` has four
  packing states.
- **Reference MC.** The archive stores a reference MC for each record; the values differ slightly between the packing
  states of a filling and are given here unchanged.
- **No time stamps.** The archive contains no acquisition times; `Record_No` gives the measurement order.
