import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import textwrap

# =========================
# 1) RUTAS A TUS ARCHIVOS
# =========================
# Ajusta estos nombres si tus CSV se llaman distinto:
NVDA_CSV = "Nvidia_stock_data.csv"   # tu dataset existente de NVIDIA
AMD_CSV  = "AMD_stock_2020_2025.csv"    # generado antes
INTC_CSV = "INTC_stock_2020_2025.csv"   # generado antes

OUT_DIR = Path("event_charts")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# 2) CARGA DE DATOS
# =========================
def load_ticker(csv_path, fallback_ticker=None):
    csv_path = Path(csv_path)

    try:
        # Intento normal: CSV “limpio” con Date en encabezado
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if "Date" in df.columns:
            # Normalizar nombres
            df.columns = [c.strip() for c in df.columns]
            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if price_col not in df.columns:
                raise ValueError("No se encuentra 'Close' ni 'Adj Close'.")
            if "Ticker" not in df.columns:
                if fallback_ticker is None:
                    fallback_ticker = csv_path.stem.split("_")[0].upper()
                df["Ticker"] = fallback_ticker
            df = df[["Date", "Ticker", "Open", "High", "Low", price_col, "Volume"]]
            df = df.rename(columns={price_col: "Price"})
            df = df.sort_values("Date").set_index("Date")
            return df
    except Exception:
        pass  # caemos al fallback si falla el parseo normal

    # Fallback para CSV con 3 filas de encabezado tipo:
    # 1) Price,Close,High,Low,Open,Volume
    # 2) Ticker,AMD,AMD,AMD,AMD,AMD
    # 3) Date,,,,,
    df = pd.read_csv(csv_path, header=None, skiprows=3,
                     names=["Date", "Close", "High", "Low", "Open", "Volume"])
    # tipos
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Close", "High", "Low", "Open", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if fallback_ticker is None:
        fallback_ticker = csv_path.stem.split("_")[0].upper()
    df["Ticker"] = fallback_ticker

    # Reordenar y renombrar
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    df = df.rename(columns={"Close": "Price"})
    df = df.sort_values("Date").set_index("Date")
    return df

nvda = load_ticker(NVDA_CSV, "NVDA")
amd  = load_ticker(AMD_CSV,  "AMD")
intc = load_ticker(INTC_CSV, "INTC")

# Alinear por fechas (no indispensable, pero útil)
# Mantendremos índices propios por ticker y trabajaremos por evento con asof.

# =========================
# 3) EVENTOS (tus hitos)
# =========================
# Para las fechas con solo mes/año, usé el día 1 del mes (puedes ajustarlo).
events = [
    # label, fecha (YYYY-MM-DD)
    ("Lanzamiento GPT-3",                      "2020-06-01"),
    ("Microsoft licencia exclusiva GPT-3",     "2021-09-01"),
    ("Lanzamiento ChatGPT",                    "2022-11-30"),
    ("Microsoft invierte 10.000M en OpenAI",   "2023-01-01"),   # ajusta si quieres el día exacto del anuncio
    ("Lanzamiento GPT-4",                      "2023-03-14"),
    ("Integración ChatGPT en Bing/Office/Azure","2023-05-01"),  # periodo mayo-julio -> tomamos 1/mayo
    ("Lanzamiento GPT-4o",                     "2024-05-13"),
]

# =========================
# 4) FUNCIONES AUXILIARES
# =========================
def baseline_price_on_or_before(df, dt):
    """Precio al cierre en la fecha del evento o el último hábil anterior."""
    # df index es Date
    if dt < df.index[0]:
        return None  # no hay datos antes del evento
    # asof: último índice <= dt
    s = df["Price"].asof(dt)
    return s

def window_series(df, dt, days_before=10, days_after=30):
    """Retorna serie de % (100 * (P/P0 - 1)) en ventana [-days_before, +days_after] hábiles."""
    start = dt - timedelta(days=days_before*2)  # ampliar para cubrir fines de semana
    end   = dt + timedelta(days=days_after*2)
    # Recorta por fechas cercanas
    sub = df.loc[(df.index >= start) & (df.index <= end), ["Price"]].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    # baseline al último hábil <= dt
    p0 = baseline_price_on_or_before(df, dt)
    if p0 is None or pd.isna(p0):
        return pd.Series(dtype=float)
    # Crear índice relativo en días hábiles
    # Re-centramos a partir del día del evento (primer índice >= dt)
    # Para visual: dejamos todos los puntos de la ventana, luego filtramos a +- días hábiles efectivos
    sub["ret_pct"] = 100.0 * (sub["Price"]/p0 - 1.0)
    # Para contar días hábiles relativos: enumerar sólo sesiones de mercado
    sessions = sub.index.sort_values()
    # Identificar posición del primer cierre >= dt (día hábil del evento)
    pos_t0 = sessions.searchsorted(pd.to_datetime(dt))
    # Calcular rango de posiciones hábiles
    lo = max(0, pos_t0 - days_before)
    hi = min(len(sessions)-1, pos_t0 + days_after)
    sub = sub.iloc[lo:hi+1].copy()
    # Reindexar con un eje relativo en sesiones: -k ... 0 ... +m
    rel_idx = list(range(lo - pos_t0, hi - pos_t0 + 1))
    return pd.Series(sub["ret_pct"].values, index=rel_idx)

def horizon_returns(df, dt, horizons=(1,5,20)):
    """Retornos % a +N días hábiles desde el evento (baseline = último hábil <= dt)."""
    if dt < df.index[0]:
        return {h: float("nan") for h in horizons}
    sessions = df.index
    base_pos = sessions.searchsorted(pd.to_datetime(dt))  # primer >= dt
    if base_pos == 0 and sessions[0] > pd.to_datetime(dt):
        # evento antes del primer dato
        return {h: float("nan") for h in horizons}
    # baseline = último <= dt
    p0 = baseline_price_on_or_before(df, pd.to_datetime(dt))
    out = {}
    for h in horizons:
        pos = base_pos + h
        if pos >= len(sessions):
            out[h] = float("nan")
            continue
        pN = df.iloc[pos]["Price"]
        out[h] = 100.0 * (pN/p0 - 1.0)
    return out

# =========================
# 5) GRAFICAR POR EVENTO
# =========================
tick_data = {
    "NVDA": nvda,
    "AMD": amd,
    "INTC": intc,
}

summary_rows = []

for label, date_str in events:
    dt = pd.to_datetime(date_str)

    # Series % relativas en ventana (-10, +30 hábiles)
    series = {}
    for tk, df in tick_data.items():
        s = window_series(df, dt, days_before=10, days_after=30)
        if not s.empty:
            series[tk] = s

    # Si no hay datos (por ejemplo, evento muy antiguo vs data desde 2021): saltar
    if not series:
        print(f"Sin datos cercanos al evento '{label}' ({date_str}) en tus CSV.")
        continue

    # Plot
    plt.figure(figsize=(9, 5.5))
    for tk, s in series.items():
        plt.plot(s.index, s.values, label=tk, linewidth=2)

    # Líneas guía y estilos mínimos
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axhline(0, linestyle=":", linewidth=1)

    title = f"{label} — Normalizado al cierre del evento (o último hábil ≤ fecha)\n{date_str}"
    # Envolver título largo
    title = "\n".join(textwrap.wrap(title, width=70))
    plt.title(title)
    plt.xlabel("Días hábiles relativos al evento (0 = día del evento)")
    plt.ylabel("Crecimiento % desde el evento")
    plt.legend()
    plt.tight_layout()

    # Guardar figura
    slug = (
        label.lower()
        .replace(" ", "-")
        .replace("ó", "o").replace("á", "a").replace("é", "e").replace("í", "i").replace("ú", "u")
        .replace("/", "-")
    )
    out_png = OUT_DIR / f"event_{date_str}_{slug}.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Figura guardada: {out_png}")

    # Tabla de retornos a horizontes
    for tk, df in tick_data.items():
        hret = horizon_returns(df, dt, horizons=(1,5,20))
        summary_rows.append({
            "Event": label,
            "EventDate": date_str,
            "Ticker": tk,
            "H+1d %": hret[1],
            "H+5d %": hret[5],
            "H+20d %": hret[20],
        })

# =========================
# 6) RESUMEN CSV
# =========================
summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "event_horizon_returns_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Resumen de retornos guardado: {summary_csv}")

# Vista rápida en consola (opcional)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(summary_df)
