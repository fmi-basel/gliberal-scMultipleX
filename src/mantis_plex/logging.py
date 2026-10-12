import logging
import sys

defaultlvl = logging.INFO
faim_hcs_defaultlvl = logging.ERROR
# logging formatters
format_without_date = logging.Formatter(
    "%(levelname)s: %(message)s"
)  # default formatter should be good for syslog / console output
format_with_date = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

mantis_plex_logger_name = "mantis_plex"
faim_hcs_logger_suffix = "faim_hcs"
_setup = False


def setup_mplex_logger():
    mplex_logger = logging.getLogger(mantis_plex_logger_name)
    mplex_logger.setLevel(defaultlvl)
    mplex_handler = logging.StreamHandler(sys.stdout)
    mplex_handler.setFormatter(format_without_date)
    mplex_logger.addHandler(mplex_handler)

    faim_hcs_logger = mplex_logger.getChild(faim_hcs_logger_suffix)
    faim_hcs_logger.setLevel(faim_hcs_defaultlvl)
    faim_hcs_logger.addHandler(mplex_handler)
    return


def setup_prefect_handlers(logger, logfile_name, clear=True):
    if clear:
        logger.handlers.clear()
    logger.setLevel(defaultlvl)
    # add a logfile with more verbose settings
    logfile_handler = logging.FileHandler(logfile_name, mode="a")
    logfile_handler.setFormatter(format_with_date)
    logger.addHandler(logfile_handler)
    # and a Stream to stdout just to handle WARNINGS and ERRORS
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(format_without_date)
    stdout_handler.setLevel(logging.WARNING)
    logger.addHandler(stdout_handler)
    return


def get_mantis_plex_logger():
    global _setup
    if not _setup:
        setup_mplex_logger()
        _setup = True
    return logging.getLogger(mantis_plex_logger_name)


def get_faim_hcs_logger():
    return get_mantis_plex_logger().getChild(faim_hcs_logger_suffix)
