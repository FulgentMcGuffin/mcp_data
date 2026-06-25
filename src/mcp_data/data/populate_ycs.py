import argparse
import os, sys, re
from collections.abc import Hashable
from datetime import datetime, date, timedelta
from typing import cast
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
import pickle
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, Path(__file__).resolve().parents[0])

# from setup_analysis import load_parquets_from_dir, get_corr_matrix
# from data_utils.sqlite_source import SQLiteSource

import numpy as np
import pandas as pd
import asyncio
import dcor
import pickle
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from PIL import Image
import imageio.v3 as iio

from plotnine import *

from mcp_data.backends import SQLiteSource


def load_parquets_from_dir(directory: str) -> pl.DataFrame:
    """Concurrently read all parquet files in *directory* and return them concatenated."""
    paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".parquet")
    ]

    async def _load_all() -> list[pl.DataFrame]:
        async def _read_parquet(path: str) -> pl.DataFrame:

            def _construct_polars(path: str) -> pl.DataFrame:
                source = os.path.basename(path).replace(".parquet", "")
                df = (
                    pl.read_parquet(path)
                    .with_columns(pl.lit(source).alias("source"))
                    .rename(
                        lambda col: (
                            f"Y{float(col):05.1f}".replace('.', 'p')  # 5 digits in case of 100Y                            
                            if re.match(r"^\d+(\.\d+)?$", str(col))
                            else col
                        )
                    )
                )
                return df

            return await asyncio.to_thread(_construct_polars, path)

        return list(await asyncio.gather(*[_read_parquet(p) for p in paths]))

    return pl.concat(asyncio.run(_load_all()))


def populate_sqlite_from_files(db_path: str):
    with SQLiteSource(db_path, read_only=False) as db:
        # Load zero rates to SQLite
        zero_rates_dir = os.getenv("ZERORATE_CURVES_DIR")
        if zero_rates_dir is None:
            raise ValueError("ZERORATE_CURVES_DIR environment variable is not set")
        db.create_table_from_polars(
            "zero_rates", load_parquets_from_dir(zero_rates_dir), True
        )
        # Load par rates to SQLite
        par_rates_dir = os.getenv("PAR_CURVES_DIR", None)
        if par_rates_dir is None:
            raise ValueError("PAR_CURVES_DIR environment variable is not set")
        db.create_table_from_polars(
            "par_rates", load_parquets_from_dir(par_rates_dir), True
        )
        # Load spot FX to SQLite
        spotfx_dir = os.getenv("SPOTFX_DIR", None)
        if spotfx_dir is None:
            raise ValueError("SPOTFX_DIR environment variable is not set")
        db.create_table_from_polars("spotfx", load_parquets_from_dir(spotfx_dir), True)
        
def get_coverage(db_path: str):
    with SQLiteSource(db_path, read_only=False) as db:
        coverage = (
            pl.concat(
                [
                    pl.DataFrame(
                        db.execute(
                            "SELECT MIN(date) AS start_date, MAX(date) AS end_date, MIN(source) AS source FROM zero_rates GROUP BY source"
                        )
                    ).with_columns(pl.lit("zero_rates").alias("type")),
                    pl.DataFrame(
                        db.execute(
                            "SELECT MIN(date) AS start_date, MAX(date) AS end_date, MIN(source) AS source FROM par_rates GROUP BY source"
                        )
                    ).with_columns(pl.lit("par_rates").alias("type")),
                    pl.DataFrame(
                        db.execute(
                            "SELECT MIN(date) AS start_date, MAX(date) AS end_date, MIN(source) AS source FROM spotfx GROUP BY source"
                        )
                    ).with_columns(pl.lit("spotfx").alias("type")),
                ]
            )
            .with_columns(
                pl.col("start_date").str.to_date(), pl.col("end_date").str.to_date()
            )
            .with_columns(
                (pl.col("end_date") - pl.col("start_date"))
                .dt.total_days()
                .alias("coverage_days")
            )
            .with_columns(pl.col("source").replace("spot_fx_rates", "fx"))
            .sort(["source", "type", "start_date", "end_date"])
        )    
    return coverage

def get_coverage_plot(coverage: pl.DataFrame, is_date_coverage: bool = True):
    if is_date_coverage:
        coverage_date_plot = (
            ggplot(
                coverage,
                aes(x="start_date", xend="end_date", y="source", yend="source", color="type"),
            )
            + geom_segment(size=5, alpha=0.5)
            + scale_x_date(date_breaks="12 months", date_labels="%b %Y")
            + labs(
                title="Date Ranges by Source and Type", x="Date", y="Source", color="Rate Type"
            )
            + scale_color_brewer(type="qual", palette="Set1")
            + theme_538()
            + theme(figure_size=(16, 5), axis_text_x=element_text(angle=45, hjust=1, size=7))
            + facet_wrap("type")
        )
        return coverage_date_plot
    else:
        coverage_days_plot = (
            ggplot(coverage, aes(x="source", y="coverage_days", fill="type")) +
            geom_bar(stat="identity", position="dodge") +
            labs(
                title="Coverage by Source and Type",
                x="Source",
                y="Coverage (days)",
                fill="Rate Type"
            ) +
            scale_fill_brewer(type="qual", palette="Set2") +
            theme_tufte() +
            theme(figure_size=(12, 3))
        )
        return coverage_days_plot
    
def df_distance_correlation_np(arr, columns):
    """
    Parameters
    ----------
    arr : np.ndarray
        Shape: (n_rows, n_columns)

    columns : list-like
        Column names corresponding to arr columns

    Returns
    -------
    pd.DataFrame
        Symmetric distance correlation matrix
    """
    n_cols = arr.shape[1]
    # preallocate output matrix
    out = np.empty((n_cols, n_cols), dtype=float)
    for i in range(n_cols):
        v_i = arr[:, i]
        # diagonal always 1
        out[i, i] = 1.0
        for j in range(i + 1, n_cols):
            v_j = arr[:, j]
            # faster NaN filtering
            mask = ~(np.isnan(v_i) | np.isnan(v_j))
            if mask.sum() == 0:
                dcor_val = np.nan
            else:
                dcor_val = dcor.distance_correlation(v_i[mask], v_j[mask])
            out[i, j] = dcor_val
            out[j, i] = dcor_val
    return pd.DataFrame(out, index=columns, columns=columns)


def melt_rolling_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """Long-form rolling correlations with one row per (date, issuer pair).

    Drops self-correlations (Var1 == Var2) and duplicate pairs by keeping
    alphabetical Var1/Var2 order per date.
    """
    melted = corr.stack().to_frame("Correlation").reset_index()
    melted.columns = ["date", "Var1", "Var2", "Correlation"]
    melted = melted[melted["Var1"] != melted["Var2"]]
    pairs = np.sort(melted[["Var1", "Var2"]].to_numpy(), axis=1)
    melted["Var1"] = pairs[:, 0]
    melted["Var2"] = pairs[:, 1]
    return (
        melted.drop_duplicates(subset=["date", "Var1", "Var2"])
        .loc[:, ["date", "Var1", "Var2", "Correlation"]]
        .astype({"Correlation": float})
    )


def to_picklable_corr_df(df: pl.DataFrame) -> pl.DataFrame:
    """Use native polars dtypes so correlation tables survive pickle round-trips."""
    if df.schema["date"] == pl.Object:
        # e.g. numpy object arrays of datetime.date cannot cast(pl.Date) directly
        df = df.with_columns(
            pl.col("date")
            .map_elements(
                lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d),
                return_dtype=pl.Utf8,
            )
            .str.to_date()
        )
    else:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df.with_columns(pl.col("Correlation").cast(pl.Float64))


def melt_distance_corr_results(
    results: np.ndarray,
    obs_dates: pd.Index,
    names: pd.Index | list[str],
) -> pl.DataFrame:
    """Long-form distance correlations with one row per (date, issuer pair).

    Emits only the upper triangle of each (N x N) matrix so Var1 <= Var2
    alphabetically, without a separate dedup pass over T * N^2 rows.
    """
    T, N, _ = results.shape
    names_arr = np.asarray(names)
    order = np.argsort(names_arr)
    names_sorted = names_arr[order]
    results = results[:, order, :][:, :, order]
    i, j = np.triu_indices(N, k=1)
    n_pairs = i.size
    date_vals = np.asarray(obs_dates, dtype="datetime64[D]")
    return to_picklable_corr_df(
        pl.DataFrame(
            {
                "date": np.repeat(date_vals, n_pairs),
                "Var1": np.tile(names_sorted[i], T),
                "Var2": np.tile(names_sorted[j], T),
                "Correlation": results[:, i, j].reshape(-1),
            }
        )
    )


def rolling_corr_matrices(
    df: pl.DataFrame,
    window: int = 252,
    tenors: list[str] | None = None,
    use_dcor: bool = False,
) -> dict[str, pd.DataFrame | pl.DataFrame]:
    """Compute per-tenor rolling inter-issuer correlation matrices.

    For every date and each requested tenor the function
    produces a square correlation matrix whose rows and columns are labelled
    by issuer (the ``source`` column).

    Parameters
    ----------
    df : pl.DataFrame
        Zero-rate curves table. Must contain at least the columns
        ``date``, ``source``, and the tenor columns named in *tenors*.
    window : int
        Rolling window expressed as a number of observations (rows).
        Default is 252, roughly one calendar year of trading days.
    tenors : list[str] or None
        Tenor column names to include.  Defaults to
        ``["Y1p0", "Y2p0", "Y5p0", "Y10p0"]``.

    Returns 
    -------
    dict[str, pd.DataFrame | pl.DataFrame]
        Keys are tenor names.  Each value is a melted table with columns
        ``date``, ``Var1``, ``Var2``, ``Correlation`` (one row per unique
        issuer pair per date, Var1 <= Var2 alphabetically).
    """
    if tenors is None:
        tenors = ["Y1p0", "Y2p0", "Y5p0", "Y10p0", "Y30p0"]

    filtered = df.sort("date")

    corr_by_tenor: dict[str, pd.DataFrame] = {}
    for tenor in tenors:
        wide_pd = (
            filtered.select(["date", "source", tenor])
            .pivot(on="source", index="date", values=tenor)
            .sort("date")
            .to_pandas()
            .set_index("date")
        )
        if not use_dcor:
            corr = (
                wide_pd.rolling(window=window, min_periods=int(window / 3)).corr()
            )
            corr.index = corr.index.set_levels(corr.index.levels[0].date, level=0)
            corr_by_tenor[tenor] = melt_rolling_corr(corr)
        else:
            arr = wide_pd.to_numpy()
            windows = sliding_window_view(
                arr,
                window_shape=(window, arr.shape[1])
            ).squeeze(1)
            results = np.array([
                df_distance_correlation_np(w, wide_pd.columns)
                for w in tqdm(windows)
            ])
            obs_dates = wide_pd.index[window - 1:]
            corr_by_tenor[tenor] = melt_distance_corr_results(
                results, obs_dates, wide_pd.columns
            )
    return corr_by_tenor

def get_corr_matrix( input_df: pl.DataFrame,
                    window: int = 65, 
                    tenors: list[str] =["Y1p0", "Y2p0", "Y5p0", "Y10p0", "Y30p0"], 
                    use_dcor:bool = True, 
                    is_polars:bool = True,):
    # Calculate correlation matrices for zero rates
    corr_matrices = rolling_corr_matrices(
        input_df,
        window=window,
        tenors=tenors,
        use_dcor=use_dcor,
    )            
    if is_polars:
        if not isinstance(corr_matrices, pl.DataFrame):
            if isinstance(corr_matrices, pd.DataFrame):
                return pl.DataFrame(corr_matrices)
            else:
                if isinstance(corr_matrices, dict):
                    return {
                        k: to_picklable_corr_df(
                            pl.from_pandas(v) if isinstance(v, pd.DataFrame) else v
                        )
                        for k, v in corr_matrices.items()
                    }
                else:
                    raise ValueError(f"Invalid type: {type(corr_matrices)}")                
    return corr_matrices
    
    
def populate_sqlite_from_files(db_path: str|None = None):
    with SQLiteSource(db_path, read_only=False) as db:
        # Load zero rates to SQLite
        LOCALDATA_PATH = os.getenv("LOCALDATA_PATH",None)
        if LOCALDATA_PATH is None:
            raise ValueError("LOCALDATA_PATH environment variable is not set")
        zero_rates_dir = os.getenv("ZERORATE_CURVES_DIR", f'{LOCALDATA_PATH}/zero_coupon')
        if zero_rates_dir is None:
            raise ValueError("ZERORATE_CURVES_DIR environment variable is not set")
        db.create_table_from_polars(
            "zero_rates", load_parquets_from_dir(zero_rates_dir), True
        )
        # Load par rates to SQLite
        par_rates_dir = os.getenv("PAR_CURVES_DIR", f'{LOCALDATA_PATH}/par_curve')
        if par_rates_dir is None:
            raise ValueError("PAR_CURVES_DIR environment variable is not set")
        db.create_table_from_polars(
            "par_rates", load_parquets_from_dir(par_rates_dir), True
        )
        # Load spot FX to SQLite
        spotfx_dir = os.getenv("SPOTFX_DIR", f'{LOCALDATA_PATH}/spot_fx_rates')
        if spotfx_dir is None:
            raise ValueError("SPOTFX_DIR environment variable is not set")
        db.create_table_from_polars("spotfx", load_parquets_from_dir(spotfx_dir), True)
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load studio data into SQLite and build or ingest correlation tables.",
    )
    parser.add_argument(
        "--load-from-files",
        action="store_true",
        default=True,
        help="Populate SQLite from parquet directories (zero rates, par, spot FX).",
    )
    parser.add_argument(
        "--create-corr-files",
        action="store_true",
        default=True,
        help="Compute correlation matrices and write melted pickle files to DATA_DIR.",
    )
    parser.add_argument(
        "--populate-sqlite-corr-from-files",
        action="store_true",
        default=False,
        help="Load melted correlation pickles and write the window_corr SQLite table.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    load_from_files = args.load_from_files
    create_corr_files = args.create_corr_files
    populate_sqlite_corr_from_files = args.populate_sqlite_corr_from_files

    if load_from_files:
        populate_sqlite_from_files()
        pass
    with SQLiteSource(read_only=False) as db:
        zero_rates = pl.DataFrame(db.execute("SELECT * FROM zero_rates")).with_columns(
            pl.col("date").str.to_date()
        )
        par_rates = pl.DataFrame(db.execute("SELECT * FROM par_rates")).with_columns(
            pl.col("date").str.to_date()
        )
        spotfx_rates = pl.DataFrame(db.execute("SELECT * FROM spotfx")).with_columns(
            pl.col("date").str.to_date()
        )

    starting_year = 2007
    starting_month = 1
    starting_dayOfMonth = 1
    
    tenors = ["Y000p5", "Y001p0", "Y002p0", "Y003p0", "Y004p0", "Y005p0", "Y007p0","Y010p0", "Y012p0", "Y015p0", "Y020p0","Y025p0", "Y030p0"]
    correlation_window_sizes = [20, 40, 60, 90]

    DEFAULT_CORR_DIR =  f"{os.getenv('DERIVED_LOCAL_DATA_PATH')}/{os.getenv('DERIVED_CORR_FOLDER')}"

    df_pl_all = []
    for input_df in [(zero_rates, "zero_rates"), (par_rates, "par_rates")]:
        input_df, input_df_name = input_df
        input_df = input_df.filter(
            pl.col("date")
            >= pl.date(starting_year, starting_month, starting_dayOfMonth)
        )
        max_date_str = input_df["date"].max().strftime("%Y%m%d")
        for correlation_window_size in correlation_window_sizes:
            corr_matrices = None
            melted_corr_file = f"{DEFAULT_CORR_DIR}/corr_dfs_melted_{input_df_name}_{starting_year:04d}{starting_month:02d}{starting_dayOfMonth:02d}_{max_date_str}_W{correlation_window_size}.pkl"
            if os.path.exists(melted_corr_file):
                print(f"Loading existing {melted_corr_file}")
                if populate_sqlite_corr_from_files:
                    corr_matrices = pickle.load(open(melted_corr_file, "rb"))
            else:
                if create_corr_files:
                    print(f"Creating {melted_corr_file}")
                    corr_matrices = get_corr_matrix(input_df, window=correlation_window_size, tenors=tenors, use_dcor=False, is_polars=True)    
                    if not os.path.exists(melted_corr_file):
                        with open(melted_corr_file, "wb") as f:
                            pickle.dump(corr_matrices, f)
            
            if populate_sqlite_corr_from_files:
                dfs_pl = [
                    df_pl.with_columns(
                        pl.lit(term).alias("observable"),
                        pl.col("Var1").alias("source1"),
                        pl.col("Var2").alias("source2"),
                        pl.lit(correlation_window_size).alias("window_size"),
                        pl.lit("spearman").alias("corr_type")
                    ).drop(["Var1", "Var2"])
                    for term, df_pl in corr_matrices.items()
                ]
                df_pl_all.append(pl.concat(dfs_pl))

            dcorr_matrices = None
            melted_dcorr_file = f"{DEFAULT_CORR_DIR}/dcorr_dfs_melted_{input_df_name}_{starting_year:04d}{starting_month:02d}{starting_dayOfMonth:02d}_{max_date_str}_W{correlation_window_size}.pkl"
            if os.path.exists(melted_dcorr_file):
                print(f"Loading existing {melted_dcorr_file}")
                if populate_sqlite_corr_from_files:
                    dcorr_matrices = pickle.load(open(melted_corr_file, "rb"))
            else:
                if create_corr_files:
                    print(f"Creating {melted_dcorr_file}")
                    dcorr_matrices = get_corr_matrix(input_df, window=correlation_window_size, tenors=tenors, use_dcor=True, is_polars=True)    
                    if not os.path.exists(melted_dcorr_file):
                        with open(melted_dcorr_file, "wb") as f:
                            pickle.dump(dcorr_matrices, f)
                            
            if populate_sqlite_corr_from_files:
                dfs_pl = [
                    df_pl.with_columns(
                        pl.lit(term).alias("observable"),
                        pl.col("Var1").alias("source1"),
                        pl.col("Var2").alias("source2"),
                        pl.lit(correlation_window_size).alias("window_size"),
                        pl.lit("dcorr").alias("corr_type")
                    ).drop(["Var1", "Var2"])
                    for term, df_pl in dcorr_matrices.items()
                ]
                df_pl_all.append(pl.concat(dfs_pl))
    
    if len(df_pl_all) > 0:
        df_pl_all = pl.concat(df_pl_all)
        df_pl_all = df_pl_all.rechunk()
        with SQLiteSource(read_only=False) as db:
            db.create_table_from_polars("window_corr", df_pl_all, True)
        print(f"Saved {df_pl_all.shape[0]} rows to SQLite in table window_corr")
        # df_pl_all.write_parquet(f"{os.getenv('DATA_DIR')}/ALL_corr_dfs_melted_{starting_year:04d}{starting_month:02d}{starting_dayOfMonth:02d}.parquet")

    print (zero_rates.shape)
    print (zero_rates[-10:])

