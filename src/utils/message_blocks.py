# utils/message_blocks.py
import inspect, os
from contextlib import contextmanager
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()

def _first_user_frame():
    cur = inspect.currentframe()
    try:
        for fi in inspect.getouterframes(cur):
            fpath = Path(fi.filename).resolve()
            if fpath.name == "contextlib.py":
                continue
            if fpath == _THIS_FILE:
                continue
            return fi.frame
        # fallback: one step back
        return cur.f_back
    finally:
        # avoid reference cycles
        del cur

def _where_str(frame):
    # make path relative to Hydra’s original cwd if available
    root = Path(os.getcwd())
    file = Path(frame.f_code.co_filename).resolve()
    rel = os.path.relpath(str(file), start=str(root.resolve()))
    func = frame.f_code.co_name
    cls = frame.f_locals.get("self", None)
    target = f"{cls.__class__.__name__}.{func}()" if cls is not None else f"{func}()"
    return f"{Path(rel).as_posix()}/{target}"

@contextmanager
def debug_block(label: str | None = None):
    frame = _first_user_frame()
    where = _where_str(frame)
    tag = label or where
    print(f"\n\n\nDEBUG [{tag}] " +
    "================================================================\n")
    try:
        yield
    except Exception as e:
        print(f"\nDEBUG [{tag}] ERROR: {e}" +
        "================================================================\n\n\n")
        raise
    finally:
        print(f"\nDEBUG [{tag}] DONE " +
        "================================================================\n\n\n")
        del frame

@contextmanager
def reply_block(label: str | None = None):
    frame = _first_user_frame()
    where = _where_str(frame)
    tag = label or where
    print(f"\n\n\nREPLY [{tag}] " +
    "================================================================\n")
    try:
        yield
    except Exception as e:
        print(f"\nREPLY [{tag}] ERROR: {e}" +
        "================================================================\n\n\n")
        raise
    finally:
        print(f"\nREPLY [{tag}] DONE " +
        "================================================================\n\n\n")
        del frame

@contextmanager
def warning_block(label: str | None = None, message: str = None):
    # frame = _first_user_frame()
    # where = _where_str(frame)
    string = ""
    if message == "failed":
        string = "\n\nFailed to measure.\n"
    # tag = label or where
    print(f"\n\n\nWARNING " +
    "================================================================\n")
    try:
        yield
    except Exception as e:
        print(f"\nWARNING ERROR: {e}" +
        "================================================================\n\n\n")
        raise
    finally:
        print(string +
              f"\nWARNING DONE " +
              "================================================================\n\n\n")
        # del frame
