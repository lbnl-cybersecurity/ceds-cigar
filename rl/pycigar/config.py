"""Default config variables, which may be overridden by a user config."""
import os.path as osp

OPENDSS_SLEEP = 1.0  # Delay between initializing OpenDSS and PyCIGAR

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

DATA_DIR = PROJECT_PATH + "/data"

LOG_DIR = PROJECT_PATH + "/result"
