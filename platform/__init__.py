"""Dynamic Multi-Agent Workflow Platform."""

# Compatibility shim: re-export stdlib platform attributes.
# This package name ('platform') shadows Python's stdlib 'platform' module
# when the project root is in sys.path. Tools like pytest call
# platform.python_version(), platform.system(), etc. internally.
import importlib.util as _importlib_util
import os.path as _osp
import sysconfig as _sysconfig


def _install_stdlib_compat() -> None:
    """Load stdlib platform.py by file path and re-export its public API."""
    stdlib_dir = _sysconfig.get_paths()["stdlib"]
    platform_py = _osp.join(stdlib_dir, "platform.py")
    if not _osp.isfile(platform_py):
        return
    spec = _importlib_util.spec_from_file_location("_stdlib_platform", platform_py)
    if not spec or not spec.loader:
        return
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    g = globals()
    for name in dir(mod):
        if not name.startswith("_") and name not in g:
            g[name] = getattr(mod, name)


_install_stdlib_compat()

del _install_stdlib_compat, _importlib_util, _osp, _sysconfig
