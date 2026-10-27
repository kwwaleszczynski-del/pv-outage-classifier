"""Event classes and their categories (shared by the simulator, the classifier and the KPIs)."""

CATEGORY = {
    "inverter_trip": "fault", "string_open": "fault", "bypass_short": "fault", "mppt_failure": "fault",
    "partial_shading": "environment", "soiling": "environment", "snow": "environment",
    "curtailment": "external", "grid_outage": "external",
    "comm_loss": "data", "frozen_data": "data", "sensor_fault": "data",
}
CLASSES = list(CATEGORY)
LOSS_CLASSES = [c for c in CLASSES if CATEGORY[c] != "data"]
DATA_CLASSES = [c for c in CLASSES if CATEGORY[c] == "data"]
