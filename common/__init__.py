from jaxtyping import install_import_hook

with install_import_hook("common", "beartype.beartype"):
    import types  # noqa: F401
