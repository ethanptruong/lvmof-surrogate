"""Service-layer wrappers around the existing CLI pipeline.

Each module in this package is a thin adapter that builds an
``argparse.Namespace`` (or equivalent) and calls into the long-standing
functions in ``main.py`` / ``data_processing.py`` / ``cosmo_features.py``.
No ML or featurization logic lives here.
"""
