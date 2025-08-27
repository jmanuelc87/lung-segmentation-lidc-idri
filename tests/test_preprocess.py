from sqlalchemy.testing.plugin.plugin_base import logging
import pytest
import logging
import pylidc as pl
import preprocess as p

from collections import deque


LOG = logging.getLogger(__name__)


def test_preprocess():
    scan = pl.query(pl.Scan).first()
    coll = deque()
    p.preprocess(scan, coll)
    LOG.info(coll)
