"""Default config variables, which may be overridden by a user config."""
import os.path as osp

OPENDSS_SLEEP = 1.0  # Delay between initializing OpenDSS and PyCIGAR

PROJECT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))

DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), 'data'))

LOG_DIR = osp.abspath(osp.join(osp.dirname(__file__), 'result'))