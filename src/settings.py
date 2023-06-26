import os

PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# useful paths to data and saved results
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results')
FILMWEB_USERNAME = None
FILMWEB_PASSWORD = None
CHROME_WEBDRIVER_PATH = None

try:
    from user_settings import *
except ImportError:
    pass
