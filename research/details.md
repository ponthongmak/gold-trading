# Model experiment: full details

Validation days (used to choose each model's settings): 2026-09-02 → 2026-09-12 (188 hourly forecasts).
Test days (never used for choosing): 2026-09-14 → 2026-10-07 (396 hourly forecasts, each 1–24 trading hours ahead).
Error = mean absolute difference between forecast and actual hourly close, in USD. Skill = how much smaller than 'no change' (positive = better).
'Better days' = on how many test days the model's average error was below 'no change'. t = paired t-statistic of those daily differences (t ≥ 2 ≈ consistently better).

## Best setting of each model (chosen on validation, scored on test)

| Model | Setting | Validation error | Test error | Test skill | Better days | t |
|---|---|---|---|---|---|---|
| ARIMA(1,1,1) | look-back 1m | $33.23 | $25.91 | +0.3% | 12/21 | 0.6 |
| No change | — | $33.23 | $25.99 | +0.0% | 0/21 | 0.0 |
| AR on returns | look-back 6m, AR(3) | $33.22 | $25.99 | -0.0% | 15/21 | 0.5 |
| Drift | 20 days | $33.26 | $26.04 | -0.2% | 12/21 | 0.4 |
| Daily pattern (current) | look-back 3m | $32.24 | $26.18 | -0.7% | 9/21 | -0.4 |
| Shrunk pattern | look-back 3m, w=1.0 | $33.09 | $26.20 | -0.8% | 3/21 | -3.6 |
| ETS | look-back 1w, damped | $30.45 | $33.70 | -29.7% | 8/21 | -1.7 |
| Seasonal naive | — | $57.23 | $39.80 | -53.2% | 1/21 | -5.0 |

## Every model and setting tried

| Model | Setting | Validation error | Validation skill | Test error | Test skill | Better days | t |
|---|---|---|---|---|---|---|---|
| AR on returns | look-back 1m, AR(3) | $33.29 | -0.2% | $25.77 | +0.8% | 14/21 | 0.8 |
| AR on returns | look-back 1m, AR(1) | $33.32 | -0.3% | $25.89 | +0.4% | 12/21 | 0.6 |
| AR on returns | look-back 1m, AR(2) | $33.30 | -0.2% | $25.90 | +0.3% | 12/21 | 0.6 |
| ARIMA(1,1,1) | look-back 1m | $33.23 | +0.0% | $25.91 | +0.3% | 12/21 | 0.6 |
| Daily pattern (current) | look-back 6m | $32.28 | +2.9% | $25.97 | +0.1% | 13/21 | 0.0 |
| ETS | look-back 3m, flat | $33.23 | +0.0% | $25.99 | +0.0% | 18/21 | 0.4 |
| ETS | look-back 6m, damped | $33.19 | +0.1% | $25.99 | +0.0% | 17/21 | 0.7 |
| ETS | look-back 6m, flat | $33.23 | +0.0% | $25.99 | +0.0% | 18/21 | 1.7 |
| No change | — | $33.23 | +0.0% | $25.99 | +0.0% | 0/21 | 0.0 |
| ETS | look-back 2w, flat | $33.23 | -0.0% | $25.99 | -0.0% | 18/21 | -0.2 |
| AR on returns | look-back 6m, AR(3) | $33.22 | +0.0% | $25.99 | -0.0% | 15/21 | 0.5 |
| ETS | look-back 1m, flat | $33.23 | +0.0% | $25.99 | -0.0% | 18/21 | -0.7 |
| Shrunk pattern | look-back 6m, w=0.25 | $33.24 | -0.0% | $25.99 | -0.0% | 9/21 | -0.0 |
| Shrunk pattern | look-back 6m, w=0.5 | $33.25 | -0.1% | $26.00 | -0.1% | 9/21 | -0.3 |
| ETS | look-back 1w, flat | $33.22 | +0.0% | $26.01 | -0.1% | 18/21 | -0.7 |
| Shrunk pattern | look-back 2w, w=0.25 | $33.35 | -0.4% | $26.01 | -0.1% | 12/21 | -0.5 |
| Shrunk pattern | look-back 6m, w=0.75 | $33.27 | -0.1% | $26.02 | -0.1% | 9/21 | -0.7 |
| Shrunk pattern | look-back 1m, w=0.25 | $33.21 | +0.1% | $26.02 | -0.1% | 11/21 | -0.6 |
| Shrunk pattern | look-back 3m, w=0.25 | $33.18 | +0.2% | $26.02 | -0.1% | 4/21 | -2.6 |
| ARIMA(1,1,1) | look-back 6m | $33.26 | -0.1% | $26.03 | -0.2% | 14/21 | 0.3 |
| Drift | 20 days | $33.26 | -0.1% | $26.04 | -0.2% | 12/21 | 0.4 |
| Shrunk pattern | look-back 6m, w=1.0 | $33.30 | -0.2% | $26.04 | -0.2% | 8/21 | -1.0 |
| AR on returns | look-back 6m, AR(2) | $33.22 | +0.0% | $26.06 | -0.3% | 13/21 | 0.1 |
| AR on returns | look-back 6m, AR(1) | $33.25 | -0.1% | $26.06 | -0.3% | 14/21 | 0.0 |
| Shrunk pattern | look-back 3m, w=0.5 | $33.14 | +0.3% | $26.07 | -0.3% | 4/21 | -3.0 |
| ETS | look-back 3m, damped | $33.37 | -0.4% | $26.08 | -0.4% | 11/21 | -1.3 |
| Shrunk pattern | look-back 1m, w=0.5 | $33.22 | +0.0% | $26.10 | -0.4% | 11/21 | -0.9 |
| Shrunk pattern | look-back 1w, w=0.25 | $33.42 | -0.6% | $26.10 | -0.4% | 12/21 | -0.4 |
| Shrunk pattern | look-back 3m, w=0.75 | $33.11 | +0.4% | $26.13 | -0.5% | 3/21 | -3.3 |
| Shrunk pattern | look-back 2w, w=0.5 | $33.57 | -1.0% | $26.14 | -0.6% | 11/21 | -1.0 |
| Daily pattern (current) | look-back 3m | $32.24 | +3.0% | $26.18 | -0.7% | 9/21 | -0.4 |
| Shrunk pattern | look-back 3m, w=1.0 | $33.09 | +0.4% | $26.20 | -0.8% | 3/21 | -3.6 |
| Shrunk pattern | look-back 1m, w=0.75 | $33.29 | -0.2% | $26.22 | -0.9% | 10/21 | -1.3 |
| Daily pattern (current) | look-back 1m | $32.74 | +1.5% | $26.24 | -1.0% | 11/21 | -0.3 |
| AR on returns | look-back 3m, AR(3) | $33.68 | -1.3% | $26.24 | -1.0% | 8/21 | -2.1 |
| AR on returns | look-back 3m, AR(2) | $33.67 | -1.3% | $26.24 | -1.0% | 7/21 | -2.2 |
| AR on returns | look-back 3m, AR(1) | $33.66 | -1.3% | $26.28 | -1.1% | 6/21 | -2.4 |
| ARIMA(1,1,1) | look-back 3m | $33.61 | -1.1% | $26.31 | -1.2% | 5/21 | -2.8 |
| Shrunk pattern | look-back 2w, w=0.75 | $33.88 | -2.0% | $26.38 | -1.5% | 11/21 | -1.4 |
| Shrunk pattern | look-back 1m, w=1.0 | $33.39 | -0.5% | $26.39 | -1.5% | 7/21 | -1.6 |
| AR on returns | look-back 2w, AR(3) | $33.83 | -1.8% | $26.43 | -1.7% | 14/21 | -0.3 |
| AR on returns | look-back 2w, AR(2) | $33.75 | -1.5% | $26.49 | -1.9% | 14/21 | -0.5 |
| AR on returns | look-back 2w, AR(1) | $33.76 | -1.6% | $26.52 | -2.0% | 13/21 | -0.6 |
| Shrunk pattern | look-back 1w, w=0.5 | $33.85 | -1.9% | $26.54 | -2.1% | 11/21 | -1.1 |
| Daily pattern (current) | look-back 2w | $32.83 | +1.2% | $26.61 | -2.4% | 8/21 | -1.1 |
| ETS | look-back 1m, damped | $34.85 | -4.9% | $26.72 | -2.8% | 8/21 | -1.8 |
| Shrunk pattern | look-back 2w, w=1.0 | $34.28 | -3.1% | $26.73 | -2.9% | 11/21 | -1.8 |
| Drift | 10 days | $33.96 | -2.2% | $26.78 | -3.1% | 11/21 | -0.8 |
| Shrunk pattern | look-back 1w, w=0.75 | $34.45 | -3.7% | $27.28 | -5.0% | 9/21 | -1.7 |
| AR on returns | look-back 1w, AR(3) | $37.61 | -13.2% | $28.15 | -8.3% | 11/21 | -1.1 |
| Shrunk pattern | look-back 1w, w=1.0 | $35.21 | -6.0% | $28.28 | -8.8% | 8/21 | -2.3 |
| AR on returns | look-back 1w, AR(1) | $37.58 | -13.1% | $28.29 | -8.9% | 8/21 | -1.4 |
| AR on returns | look-back 1w, AR(2) | $37.48 | -12.8% | $28.31 | -8.9% | 10/21 | -1.4 |
| Daily pattern (current) | look-back 1w | $33.99 | -2.3% | $28.40 | -9.3% | 8/21 | -2.1 |
| Drift | 5 days | $37.48 | -12.8% | $28.52 | -9.7% | 9/21 | -1.6 |
| ETS | look-back 1w, damped | $30.45 | +8.4% | $33.70 | -29.7% | 8/21 | -1.7 |
| Seasonal naive | — | $57.23 | -72.2% | $39.80 | -53.2% | 1/21 | -5.0 |
| ETS | look-back 2w, damped | $34.20 | -2.9% | $46.31 | -78.2% | 10/21 | -1.2 |
| ETS | look-back 1m, trend | $102.81 | -209.4% | $108.31 | -316.8% | 2/21 | -4.9 |
| ETS | look-back 3m, trend | $73.48 | -121.1% | $150.41 | -478.8% | 4/21 | -1.8 |
| ETS | look-back 1w, trend | $74.98 | -125.6% | $156.75 | -503.2% | 3/21 | -2.3 |
| ETS | look-back 2w, trend | $76.69 | -130.8% | $162.98 | -527.2% | 3/21 | -2.3 |
| ETS | look-back 6m, trend | $309.42 | -831.1% | $235.70 | -807.0% | 1/21 | -4.9 |

Overfitting example: ETS (look-back 1w, damped) looked best on the validation days but was −29.7% on the test days. Small wins on a few days often do not repeat.
