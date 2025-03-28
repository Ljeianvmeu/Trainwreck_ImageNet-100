"""
commons.py

Common functionality and variable constants used across the entire SW package.
"""

import configparser
from datetime import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.ini")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
ATTACK_DATA_DIR_REL = "attack_data"
ATTACK_DATA_DIR = os.path.join(SCRIPT_DIR, ATTACK_DATA_DIR_REL)
ATTACK_INSTRUCTIONS_DIR = os.path.join(ATTACK_DATA_DIR, "instructions")
ATTACK_FEAT_REPR_DIR = os.path.join(ATTACK_DATA_DIR, "feat_repre")
JSD_MAT_DIR = os.path.join(SCRIPT_DIR, "jsd_matrices")
ATTACK_TRAINWRECK_DIR = os.path.join(ATTACK_DATA_DIR, "trainwreck")
ATTACK_TW_POISONED_DATA_DIR = os.path.join(ATTACK_TRAINWRECK_DIR, "poisoned_data")
PROJECT_DIR = 'trainwreck_imagenet100'

# Load the data root dir from config.ini
cfg = configparser.ConfigParser()
# If the autodl-tmp directory exists in the environment, use a specific path
if os.path.exists(os.path.join(PROJECT_DIR, 'data')):
    # ImageNet-100 data path
    ROOT_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
else:
    # Create data directory
    ROOT_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    os.makedirs(ROOT_DATA_DIR, exist_ok=True)

if not os.path.isabs(ROOT_DATA_DIR):
    ROOT_DATA_DIR = os.path.join(SCRIPT_DIR, ROOT_DATA_DIR)

if not os.path.exists(ROOT_DATA_DIR):
    os.makedirs(ROOT_DATA_DIR)


def timestamp() -> str:
    """
    A timestamp for the SW package's console output.
    """
    return f"[{datetime.now().strftime('%d %b %Y, %H:%M:%S')}]"


def t_readable(seconds):
    """
    Converts time in seconds to a more easily-readable format of days, hours,
    minutes, and seconds.
    """
    days = int(seconds // (24 * 60 * 60))
    seconds %= 24 * 60 * 60

    hours = int(seconds // (60 * 60))
    seconds %= 60 * 60

    minutes = int(seconds // 60)
    seconds = round(seconds % 60, 2)

    if days > 0:
        tstamp = f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
    elif hours > 0:
        tstamp = f"{hours} hours, {minutes} minutes, {seconds} seconds"
    elif minutes > 0:
        tstamp = f"{minutes} minutes, {seconds} seconds"
    else:
        tstamp = f"{seconds} seconds"

    return tstamp