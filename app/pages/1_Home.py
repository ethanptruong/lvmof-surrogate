"""Home / status dashboard.

Mirrors the streamlit_app.py landing screen so the chemist can navigate
back here from any page.  See the entry point for layout details.
"""

from __future__ import annotations

import os
import sys

# Same path bootstrap as streamlit_app.py — pages run as standalone scripts.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import streamlit as st  # noqa: E402

from app.services.status import (dataset_summary, iteration_history,
                                  model_status)  # noqa: E402
from app.ui.components import (about_panel, metric_card, page_header,
                                status_pill)  # noqa: E402

st.set_page_config(page_title="Home \u00b7 COMPASS",
                   layout="wide")

page_header(
    "COMPASS \u00b7 Home",
    caption=("**C**losed-loop **O**ptimization of **M**OF **P**rocess "
             "**a**nd **S**olvent **S**ynthesis"),
)

about_panel()

dataset = dataset_summary()
history = iteration_history()
model   = model_status()

c1, c2 = st.columns(2)
with c1:
    metric_card(
        "Experiments recorded",
        f"{dataset.n_experiments:,}",
        sub=(f"updated {dataset.last_modified:%Y-%m-%d %H:%M}"
             if dataset.last_modified else ""),
    )
with c2:
    metric_card(
        "BO iterations run",
        f"{history.iteration}",
        sub=(f"last run {history.last_run_mtime:%Y-%m-%d %H:%M}"
             if history.last_run_mtime else "no runs yet"),
    )

st.write("")
if model.tuned:
    status_pill(
        f"Model tuned ({model.tuned_at:%Y-%m-%d})" if model.tuned_at
        else "Model tuned",
        ok=True,
    )
else:
    status_pill("Model not yet tuned - using default hyperparameters", ok=False)

st.divider()

b1, b2 = st.columns(2)
with b1:
    if st.button("Get recommendations", type="primary",
                 width="stretch"):
        try:
            st.switch_page("pages/2_Recommend.py")
        except Exception:
            st.info("Use the sidebar to open the **Recommend** page.")
with b2:
    if st.button("Record an experiment result",
                 width="stretch"):
        try:
            st.switch_page("pages/3_Record_Result.py")
        except Exception:
            st.info("Use the sidebar to open the **Record Result** page.")
