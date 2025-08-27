import pytest
import pylidc as pl
import preprocess as p


def test_preprocess():
    scan = pl.query(pl.Scan).first()
    p.preprocess(scan)
