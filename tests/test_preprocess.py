import pytest
import logging
import pylidc as pl
import preprocess as p

from tqdm import tqdm
from collections import deque


LOG = logging.getLogger(__name__)


@pytest.mark.skip()
def test_preprocess():
    scan = pl.query(pl.Scan).first()
    coll = deque()
    prog = tqdm()
    p.preprocess(scan, coll, prog)
    LOG.info(coll)
