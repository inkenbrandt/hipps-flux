import pytest
from dataclasses import dataclass
from easyfluxpy.data_quality import StationarityTest

@dataclass
class TestStationarityTest:
    
    def test_RN_uw(self):
        stationarity = StationarityTest(RN_uw=0.5, RN_wT=0.5, RN_wq=0.5, RN_wc=0.5)

        assert isinstance(stationarity.RN_uw, float)
        assert stationarity.RN_uw >= 0
    
    def test_RN_wT(self):
        stationarity = StationarityTest(RN_uw=0.5, RN_wT=0.5, RN_wq=0.5, RN_wc=0.5)

        assert isinstance(stationarity.RN_wT, float)
        assert stationarity.RN_wT >= 0
   
    def test_RN_wq(self):
        stationarity = StationarityTest(RN_uw=0.5, RN_wT=0.5, RN_wq=0.5, RN_wc=0.5)

        assert isinstance(stationarity.RN_wq, float)
        assert stationarity.RN_wq >= 0

    def test_RN_wc(self):
        stationarity = StationarityTest(RN_uw=0.5, RN_wT=0.5, RN_wq=0.5, RN_wc=0.5)

        assert isinstance(stationarity.RN_wc, float)
        assert stationarity.RN_wc >= 0