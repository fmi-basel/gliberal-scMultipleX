import logging

defaultlvl = logging.INFO
# logging formatters
format_without_date = logging.Formatter('%(levelname)s: %(message)s') # default formatter should be good for syslog / console output
format_with_date = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

def setup_prefect_handlers(logger, logfile_name, clear = True):
	if clear:
		logger.handlers.clear()
	logger.setLevel(defaultlvl)
	logfile_handler = logging.FileHandler(logfile_name, mode='a')
	logfile_handler.setFormatter(format_with_date)
	logger.addHandler(logfile_handler)
	return
