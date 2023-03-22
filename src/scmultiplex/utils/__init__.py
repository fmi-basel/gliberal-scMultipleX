import os

def get_core_count():
	'''get number of cores available to run tasks'''
	try:
		#NOTE: only available on some Unix platforms
		return len(os.sched_getaffinity(0))
	except AttributeError:
		return os.cpu_count()
