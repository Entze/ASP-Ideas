from typing import Optional, Iterable


def typecheck(value, expected, attribute: Optional[str] = None):
    if attribute:
        assert getattr(value, attribute) is expected, "{} {} should have type {}, but has type {}.".format(
            type(value).__name__, value, expected, getattr(value, attribute))
    elif expected is None:
        assert value is expected, "{} {} should have type {}, but has type {}.".format(
            type(value).__name__, value, expected, type(value).__name__)
    else:
        assert isinstance(value, expected), "{} {} should have type {}, but has type {}.".format(
            type(value).__name__, value, expected, type(value).__name__)
    return True


def open_join_close(on: str, open: str, close: str, iterable: Optional[Iterable[str]] = None):
    if iterable is not None:
        joined = on.join(iterable)
        if len(joined) > 0:
            return "{}{}{}".format(open, joined, close)
    return ""


def join_wrap(on: str, wrap_open: str, wrap_close: str, iterable: Optional[Iterable[str]] = None):
    if iterable is not None and iterable:
        return on.join("{}{}{}".format(wrap_open, it, wrap_close) for it in iterable)


def open_join_close_wrap(on: str, open: str, close: str, wrap_open: str, wrap_close: str,
                         iterable: Optional[Iterable[str]] = None):
    if iterable is not None and iterable:
        return "{}{}{}".format(open, on.join("{}{}{}".format(wrap_open, it, wrap_close) for it in iterable), close)
    return ""
