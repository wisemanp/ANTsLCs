import os
import pandas as pd
from typing import Dict


def _to_hdf5_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df coerced to HDF5-friendly dtypes (no Arrow-backed dtypes).

    - Columns converted to plain Python str
    - Index converted to numpy-backed (object/float) types
    - Values coerced to numeric float where possible
    """
    out = df.copy()
    # Ensure plain Python string column labels
    try:
        out.columns = [str(c) for c in out.columns]
    except Exception:
        # Fallback to object dtype
        out.columns = out.columns.astype(object)

    # Ensure numpy-backed index
    if isinstance(out.index, pd.MultiIndex):
        # Convert each level to object to avoid Arrow dtypes
        out.index = pd.MultiIndex.from_tuples(
            [tuple(str(x) if not hasattr(x, 'item') else x.item() for x in t) for t in out.index.tolist()],
            names=out.index.names,
        )
    else:
        try:
            # Keep numeric index if possible; else cast to object strings
            if pd.api.types.is_numeric_dtype(out.index.dtype):
                out.index = pd.Index(out.index.astype('float64'), name=out.index.name)
            else:
                out.index = pd.Index([str(x) for x in out.index.tolist()], name=out.index.name, dtype=object)
        except Exception:
            out.index = pd.Index([str(x) for x in out.index.tolist()], name=out.index.name, dtype=object)

    # Coerce values to numeric float64 where possible
    out = out.apply(pd.to_numeric, errors='coerce')
    try:
        out = out.astype('float64')
    except Exception:
        # Mixed blocks are okay; HDF5 will store as table if needed
        pass
    return out


def _to_hdf5_safe_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce metadata DataFrame to HDF5-friendly types (no Arrow-backed dtypes)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col].dtype):
            out[col] = out[col].astype(object)
        elif pd.api.types.is_integer_dtype(out[col].dtype) or pd.api.types.is_float_dtype(out[col].dtype):
            # Cast to numpy dtypes
            try:
                out[col] = pd.to_numeric(out[col], errors='coerce')
            except Exception:
                pass
        else:
            # Fallback: store as object string representation
            try:
                out[col] = out[col].astype(object)
            except Exception:
                out[col] = out[col].apply(lambda x: str(x))
    return out


def save_lightcurves_hdf5(dset, sources, output_file: str, band_F_density_conversion_dict: Dict[str, float], include_meta: bool = True) -> str:
    """
    Save simulated light curves to a single HDF5 file with multiple keys.

    For each simulated object (by index in dset.obs_index), four DataFrames are saved:
    - obj_<i>/flux: rows=MJD, columns=band, values=flux
    - obj_<i>/fluxerr: rows=MJD, columns=band, values=fluxerr
    - obj_<i>/flux_density: rows=MJD, columns=band, values=flux density (erg s^-1 cm^-2 Å^-1)
    - obj_<i>/flux_density_err: rows=MJD, columns=band, values=flux density error
    Optionally also save metadata at obj_<i>/meta (z, ra, dec, etc.).

    Returns the absolute path to the written HDF5 file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    store_path = os.path.abspath(output_file)

    with pd.HDFStore(store_path, mode="w") as store:
        for i, index in enumerate(dset.obs_index):
            obs = dset.get_target_lightcurve(index=index).copy()
            # Flux and error (AB system scaled units)
            df_flux = (
                obs.pivot_table(index="mjd", columns="band", values="flux", aggfunc="first")
                .sort_index()
            )
            df_fluxerr = (
                obs.pivot_table(index="mjd", columns="band", values="fluxerr", aggfunc="first")
                .sort_index()
            )

            # Flux density and error
            obs_fd = obs.copy()
            conv = obs_fd["band"].map(band_F_density_conversion_dict)
            obs_fd["flux_density"] = obs_fd["flux"] * conv
            obs_fd["flux_density_err"] = obs_fd["fluxerr"] * conv

            df_flux_density = (
                obs_fd.pivot_table(index="mjd", columns="band", values="flux_density", aggfunc="first")
                .sort_index()
            )
            df_flux_density_err = (
                obs_fd.pivot_table(index="mjd", columns="band", values="flux_density_err", aggfunc="first")
                .sort_index()
            )

            key_prefix = f"obj_{i}"
            # Coerce to HDF5-safe frames
            df_flux = _to_hdf5_safe_df(df_flux)
            df_fluxerr = _to_hdf5_safe_df(df_fluxerr)
            df_flux_density = _to_hdf5_safe_df(df_flux_density)
            df_flux_density_err = _to_hdf5_safe_df(df_flux_density_err)

            store.put(f"{key_prefix}/flux", df_flux, format="fixed")
            store.put(f"{key_prefix}/fluxerr", df_fluxerr, format="fixed")
            store.put(f"{key_prefix}/flux_density", df_flux_density, format="fixed")
            store.put(f"{key_prefix}/flux_density_err", df_flux_density_err, format="fixed")

            if include_meta:
                meta = sources.data.iloc[index, :]
                meta_df = pd.DataFrame(meta).T
                # Convert any object dtypes that are actually numeric to float64
                for col in meta_df.columns:
                    if meta_df[col].dtype == object:
                        # Try numeric
                        coerced = pd.to_numeric(meta_df[col], errors='coerce')
                        if coerced.notna().all():
                            meta_df[col] = coerced.astype('float64')
                        else:
                            # Fall back to string objects
                            meta_df[col] = meta_df[col].astype(object)
                meta_df = _to_hdf5_safe_meta(meta_df)
                # Use fixed format to avoid table schema inference issues
                store.put(f"{key_prefix}/meta", meta_df, format="fixed")

    return store_path
