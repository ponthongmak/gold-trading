# Gold forecast model comparison (real Twelve Data XAU/USD, 5-minute bars)

Data: 2026-02-23 16:35:00 → 2026-10-07 10:35:00 Thai time, 44,653 trading bars (20,347 closed-market bars removed: weekends and the daily break).
Forecasts: every trading hour, 1–24 trading hours ahead, using only earlier data. Look-back windows tried: 1w, 2w, 1m, 3m, 6m (max 6 months). Settings chosen on the 10 trading days before each test window.

### 5–7 Oct 2026 (every hourly forecast from 5 Oct 00:00)

| Rank | Model | Settings (chosen on validation) | MAE $ | MAE % | RMSE $ | MAE 1h / 4h / 12h / 24h $ | Direction 4h / 24h | vs no change |
|---|---|---|---|---|---|---|---|---|
| 1 | ETS | look-back 6m, damped | 16.85 | 0.39% | 20.52 | 7.6 / 14.0 / 17.6 / 16.8 | 62% / 68% | +0.0% |
| 2 | No change (naive) | — | 16.85 | 0.39% | 20.52 | 7.6 / 14.0 / 17.6 / 16.8 | — / — | +0.0% |
| 3 | Ensemble (median) | naive + shrunk + ETS + AR | 16.88 | 0.39% | 20.61 | 7.6 / 13.8 / 17.8 / 16.7 | 62% / 54% | -0.2% |
| 4 | Shrunk pattern | look-back 2w, w=0.25 | 16.98 | 0.40% | 20.71 | 7.8 / 13.4 / 18.5 / 16.5 | 58% / 46% | -0.8% |
| 5 | Daily pattern (current) | look-back 6m | 17.54 | 0.41% | 22.03 | 8.0 / 15.1 / 15.6 / 21.0 | 58% / 43% | -4.1% |
| 6 | AR on returns | look-back 1m, AR(3) | 17.87 | 0.42% | 22.61 | 7.6 / 13.4 / 18.3 / 22.0 | 56% / 43% | -6.1% |
| 7 | ARIMA(1,1,1) | look-back 1m | 18.37 | 0.43% | 23.24 | 7.5 / 14.0 / 18.5 / 22.1 | 54% / 43% | -9.1% |
| 8 | Drift | 20 days | 18.42 | 0.43% | 23.25 | 7.5 / 14.0 / 18.6 / 21.7 | 54% / 43% | -9.3% |
| 9 | Seasonal naive | — | 21.92 | 0.51% | 27.62 | 12.1 / 18.2 / 23.9 / 29.4 | 56% / 50% | -30.2% |

### Last 20 trading days (robustness check)

| Rank | Model | Settings (chosen on validation) | MAE $ | MAE % | RMSE $ | MAE 1h / 4h / 12h / 24h $ | Direction 4h / 24h | vs no change |
|---|---|---|---|---|---|---|---|---|
| 1 | ARIMA(1,1,1) | look-back 1m | 25.91 | 0.60% | 35.40 | 8.6 / 17.0 / 28.0 / 33.6 | 51% / 58% | +0.3% |
| 2 | No change (naive) | — | 25.99 | 0.60% | 36.16 | 8.6 / 17.0 / 28.0 / 34.3 | — / — | +0.0% |
| 3 | AR on returns | look-back 6m, AR(3) | 25.99 | 0.61% | 36.16 | 8.6 / 16.9 / 28.0 / 34.3 | 51% / 56% | -0.0% |
| 4 | Drift | 20 days | 26.04 | 0.61% | 35.76 | 8.6 / 17.1 / 28.1 / 33.7 | 50% / 59% | -0.2% |
| 5 | Ensemble (median) | naive + shrunk + ETS + AR | 26.05 | 0.61% | 36.19 | 8.6 / 17.0 / 28.1 / 34.4 | 49% / 51% | -0.3% |
| 6 | Daily pattern (current) | look-back 3m | 26.18 | 0.61% | 36.31 | 9.0 / 17.7 / 28.2 / 34.3 | 52% / 54% | -0.7% |
| 7 | Shrunk pattern | look-back 3m, w=1.0 | 26.20 | 0.61% | 36.32 | 8.9 / 17.2 / 28.4 / 34.4 | 52% / 47% | -0.8% |
| 8 | ETS | look-back 1w, damped | 33.70 | 0.78% | 47.89 | 9.0 / 19.1 / 33.8 / 51.0 | 45% / 47% | -29.7% |
| 9 | Seasonal naive | — | 39.80 | 0.93% | 53.67 | 12.1 / 23.5 / 41.0 / 58.3 | 53% / 41% | -53.2% |
