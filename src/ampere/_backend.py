"""
Ampere backend registry.

Call set_backend('pandas') or set_backend('arkouda') to switch at runtime.
The `ak` proxy object always forwards to the currently active backend, so
existing code that does `from ._backend import ak` needs no further changes.
"""

_backend_name: str = ''
_backend_module = None


class _AkProxy:
    """Proxy that forwards attribute access to the active backend."""

    def __getattr__(self, name):
        if _backend_module is None:
            raise RuntimeError("Ampere backend not initialised — call set_backend() first.")
        return getattr(_backend_module, name)

    def __repr__(self):
        return f"<AmpereBackendProxy: {_backend_name!r}>"


ak = _AkProxy()


def set_backend(name: str) -> None:
    """
    Switch the active backend.

    Args:
        name: 'arkouda' or 'pandas'
    """
    global _backend_name, _backend_module
    if name == 'arkouda':
        import arkouda as _ak
        _backend_module = _ak
        _backend_name = 'arkouda'
    elif name == 'pandas':
        from ._pandas_backend import PandasBackend
        _backend_module = PandasBackend
        _backend_name = 'pandas'
    else:
        raise ValueError(f"Unknown backend {name!r}. Choose 'arkouda' or 'pandas'.")


def get_backend() -> str:
    """Return the name of the currently active backend ('arkouda' or 'pandas')."""
    return _backend_name


def _init_default() -> None:
    global _backend_name, _backend_module
    try:
        import arkouda as _ak
        _backend_module = _ak
        _backend_name = 'arkouda'
    except Exception:
        from ._pandas_backend import PandasBackend
        _backend_module = PandasBackend
        _backend_name = 'pandas'


_init_default()
