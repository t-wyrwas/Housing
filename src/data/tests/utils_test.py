import numpy as np
import pandas as pd
import data.utils as utils


def test_utils_get_timestamp_returns_string_of_certain_format():
    now = utils._get_timestamp()
    assert type(now) == str
    assert len(now.split('_')) == 4
