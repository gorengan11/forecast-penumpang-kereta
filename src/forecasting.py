"""Fungsi bersama untuk proyek peramalan jumlah penumpang kereta api.

Semua notebook memakai modul ini supaya cara memuat data, membuat fitur,
dan menghitung metrik selalu sama di setiap model.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES = ROOT / "images"

# Bulan Idulfitri (Lebaran) 2016-2023, puncak arus mudik.
LEBARAN_MONTHS = {
    "2016-07", "2017-06", "2018-06", "2019-06",
    "2020-05", "2021-05", "2022-05", "2023-04",
}

# Periode pembatasan mobilitas COVID-19 (PSBB/PPKM).
COVID_START, COVID_END = "2020-04", "2021-12"


def quiet():
    """Matikan log dan peringatan yang tidak relevan dari Prophet/statsmodels."""
    warnings.filterwarnings("ignore")
    for name in ("prophet", "prophet.plot", "cmdstanpy"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True


def load_passengers() -> pd.Series:
    """Data bulanan 2016-2023 sebagai Series dengan indeks akhir bulan."""
    df = pd.read_excel(DATA_RAW / "jumlah_penumpang_2016_2023.xlsx")
    idx = pd.to_datetime(df["date"], format="%YM%m") + pd.offsets.MonthEnd(0)
    s = pd.Series(df["passenger_count"].to_numpy(), index=idx, name="penumpang")
    s.index.freq = "ME"
    return s


def load_holidays() -> pd.DataFrame:
    h = pd.read_csv(DATA_RAW / "hari_libur_2016_2023.csv", parse_dates=["tanggal"])
    return h.drop_duplicates("tanggal")


def make_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fitur eksternal per bulan.

    - jumlah_libur : banyaknya hari libur nasional + cuti bersama di bulan itu.
      (Versi skripsi mencocokkan tanggal akhir bulan dengan tanggal libur harian,
      sehingga fiturnya hampir selalu 0.)
    - lebaran      : 1 jika bulan itu bulan Idulfitri.
    - covid        : 1 selama periode pembatasan mobilitas.
    """
    h = load_holidays()
    per_month = h.groupby(h["tanggal"].dt.to_period("M")).size()
    period = index.to_period("M")
    return pd.DataFrame(
        {
            "jumlah_libur": per_month.reindex(period, fill_value=0).to_numpy(),
            "lebaran": period.strftime("%Y-%m").isin(LEBARAN_MONTHS).astype(int),
            "covid": ((period >= COVID_START) & (period <= COVID_END)).astype(int),
        },
        index=index,
    )


def metrics(actual, forecast) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    err = actual - forecast
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAPE (%)": np.mean(np.abs(err / actual)) * 100,
    }


def metrics_table(actual, forecasts: dict) -> pd.DataFrame:
    rows = {name: metrics(actual, f) for name, f in forecasts.items()}
    return pd.DataFrame(rows).T.sort_values("MAPE (%)")


# ----------------------------------------------------------------- plot ---

COLORS = {
    "Aktual": "#0b0b0b",
    "SARIMAX": "#2a78d6",
    "Prophet": "#eb6834",
    "Hybrid": "#1baf7a",
    "Seasonal naive": "#8a8984",
}


def setup_plot():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "lines.linewidth": 2,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def million_axis(ax):
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f} jt"))


# ---------------------------------------------------------------- model ---

def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Ramalan = nilai bulan yang sama setahun sebelumnya."""
    last_year = train.iloc[-12:].to_numpy()
    return np.resize(last_year, horizon)


def fit_sarimax(train: pd.Series, order, seasonal_order, exog=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def fit_predict_prophet(train: pd.Series, horizon: int, *,
                        regressors=("covid", "lebaran"),
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0):
    """Latih Prophet pada data bulanan lalu ramal `horizon` bulan ke depan.

    Nilai bawaan parameter = nilai default library Prophet, sengaja tidak di-tuning
    (lihat README bagian metodologi).
    """
    from prophet import Prophet

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    for r in regressors:
        model.add_regressor(r)

    df = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
    exog = make_exog(train.index)
    for r in regressors:
        df[r] = exog[r].to_numpy()
    model.fit(df)

    future_idx = pd.date_range(train.index[-1], periods=horizon + 1, freq="ME")[1:]
    future = pd.DataFrame({"ds": future_idx})
    exog_f = make_exog(future_idx)
    for r in regressors:
        future[r] = exog_f[r].to_numpy()
    return model.predict(future)["yhat"].to_numpy(), model
