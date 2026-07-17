"""Streamlit GUI (Phase 3).

Entry point ``gui/app.py`` (launch with ``cge gui`` or ``streamlit run``). Pages under
``gui/pages`` render via the ``GuiService`` façade; ``service.py`` and ``results_view.py``
are Streamlit-free so their logic is unit-tested without a running server.
"""

APP_PATH = __path__[0] + "/app.py"
