import numpy as np

import matplotlib as plt
import subprocess


def get_phodcos_path():
    phodcos_path = subprocess.run(
        "echo $PHODCOS_PATH", shell=True, capture_output=True, text=True
    ).stdout.strip("\n")
    return phodcos_path
