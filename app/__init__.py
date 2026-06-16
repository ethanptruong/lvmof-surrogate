"""LVMOF-Surrogate chemist-facing Streamlit application.

A thin GUI layer on top of the existing CLI pipeline.  All long-running work
(featurization, BO, retune) is delegated to the existing functions in
``main.py``, ``bo_core.py`` and ``data_processing.py`` - this package only
adds presentation, input validation, and safe Excel I/O.
"""
