import logging
import sys

defaultlvl = logging.INFO
# logging formatters
format_without_date = logging.Formatter('%(levelname)s: %(message)s') # default formatter should be good for syslog / console output
format_with_date = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

scmp_logger_name = 'scmultiplex'
_setup = False

def setup_scmp_logger():
	scmp_logger = logging.getLogger(scmp_logger_name)
	scmp_logger.setLevel(defaultlvl)
	scmp_handler = logging.StreamHandler(sys.stdout)
	scmp_logger.addHandler(scmp_handler)
	return

def setup_prefect_handlers(logger, logfile_name, clear = True):
	if clear:
		logger.handlers.clear()
	logger.setLevel(defaultlvl)
	logfile_handler = logging.FileHandler(logfile_name, mode='a')
	logfile_handler.setFormatter(format_with_date)
	logger.addHandler(logfile_handler)
	return

def get_scmultiplex_logger():
	global _setup
	if not _setup:
		setup_scmp_logger()
		_setup = True
	return logging.getLogger(scmp_logger_name)
