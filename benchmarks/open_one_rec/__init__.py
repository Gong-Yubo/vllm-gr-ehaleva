try:
    from benchmarks.open_one_rec import one_rec_patch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    one_rec_patch = None  # type: ignore
