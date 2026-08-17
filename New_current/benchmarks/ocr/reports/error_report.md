# OCR Confusion and Correction Report

All shown text is synthetic and de-identified.

## OCR mistakes

- pdf: No approved ground-truth corpus was supplied; OCR mistakes cannot be scored.

## Regex fixes

- Corrections: 11; latency: 0.047 ms
  - Before: `BP 120 / 80  haemoglobn 6 . 5 % Metformin`
  - After: `BP 120 / 80 haemoglobn 6.5% Metformin`

## Medical abbreviation fixes

- Corrections: 1; latency: 0.022 ms
  - Before: `BP 120 / 80 haemoglobn 6.5% Metformin`
  - After: `blood pressure (BP) 120 / 80 haemoglobn 6.5% Metformin`

## SymSpell corrections

- Corrections: 1; latency: 0.147 ms
  - Before: `blood pressure (BP) 120 / 80 haemoglobn 6.5% Metformin`
  - After: `blood pressure (BP) 120 / 80 hemoglobin 6.5% Metformin`
