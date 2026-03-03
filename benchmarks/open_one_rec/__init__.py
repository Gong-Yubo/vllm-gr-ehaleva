try:
    from . import one_rec_patch

    print("OpenOneRec patch applied.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    one_rec_patch = None  # type: ignore
    print("OpenOneRec patch NOT applied.")
