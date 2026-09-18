import contextlib
import functools
import importlib.metadata
import importlib.util
import io
import logging
import os
import pathlib
import sys

_FILTER_ENV = "DIFFUSION_POLICY_FILTER_SPAWN_IMPORT_NOTICES"

_ROBOSUITE_MODELS_WARNING = (
    "Could not import robosuite_models. Some robots may not be available. "
    "If you want to use these robots, please install robosuite_models from "
    "source (https://github.com/ARISE-Initiative/robosuite_models) or through "
    "pip install."
)
_ROBOSUITE_MINK_WARNING = (
    "Could not load the mink-based whole-body IK. Make sure you install "
    "related import properly, otherwise you will not be able to use the "
    "default IK controller setting for GR1 robot."
)
_ROBOSUITE_MACRO_WARNINGS = {
    "No private macro file found!",
    "It is recommended to use a private macro file",
}
_ROBOMIMIC_DATASET_VERSION_WARNING = (
    "No environment version found in dataset!\n"
    "Cannot verify if dataset and installed environment versions match"
)


def worker_import_filter_enabled():
    return os.environ.get(_FILTER_ENV) == "1"


def _package_setup_message(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return None
    package_dir = pathlib.Path(spec.origin).parent
    return f"To setup, run: python {package_dir}/scripts/setup_macros.py"


def _gym_notice():
    try:
        version = importlib.metadata.version("gym")
        from gym_notices.notices import notices

        return notices.get(version)
    except Exception:
        return None


@contextlib.contextmanager
def filter_gym_notice_for_marked_worker():
    if not worker_import_filter_enabled():
        yield
        return

    original_stderr = sys.stderr
    captured = io.StringIO()
    sys.stderr = captured
    try:
        yield
    finally:
        sys.stderr = original_stderr
        output = captured.getvalue()
        notice = _gym_notice()
        if notice is not None:
            output = output.replace(f"{notice}\n", "", 1)
        if output:
            original_stderr.write(output)
            original_stderr.flush()


class _ExactRobosuiteImportFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        messages = {
            _ROBOSUITE_MODELS_WARNING,
            _ROBOSUITE_MINK_WARNING,
            *_ROBOSUITE_MACRO_WARNINGS,
        }
        setup_message = _package_setup_message("robosuite")
        if setup_message is not None:
            messages.add(setup_message)
        self.messages = frozenset(messages)

    def filter(self, record):
        return record.getMessage() not in self.messages


def configure_robosuite_logging():
    # Robosuite installs its own console handler. Propagating the same record to
    # Hydra's root handler prints every warning twice in the parent process.
    logging.getLogger("robosuite_logs").propagate = False


def _install_robosuite_filter():
    logger = logging.getLogger("robosuite_logs")
    if any(isinstance(item, _ExactRobosuiteImportFilter) for item in logger.filters):
        return
    logger.addFilter(_ExactRobosuiteImportFilter())


def _install_robomimic_filter():
    from robomimic.utils import log_utils

    if getattr(log_utils.log_warning, "_filters_spawn_import_notices", False):
        return

    ignored_messages = {_ROBOMIMIC_DATASET_VERSION_WARNING}
    setup_message = _package_setup_message("robomimic")
    ignored_messages.add(
        "No private macro file found!"
        "\nIt is recommended to use a private macro file"
        f"\n{setup_message}"
    )
    original = log_utils.log_warning

    @functools.wraps(original)
    def filtered_log_warning(message, *args, **kwargs):
        if message in ignored_messages:
            return None
        return original(message, *args, **kwargs)

    filtered_log_warning._filters_spawn_import_notices = True
    log_utils.log_warning = filtered_log_warning


def install_marked_worker_import_filters():
    if not worker_import_filter_enabled():
        return
    _install_robosuite_filter()
    _install_robomimic_filter()


@contextlib.contextmanager
def mark_spawn_import_filter(enabled):
    if not enabled:
        yield
        return

    previous = os.environ.get(_FILTER_ENV)
    os.environ[_FILTER_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_FILTER_ENV, None)
        else:
            os.environ[_FILTER_ENV] = previous
