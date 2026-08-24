from turkiye_grid_fm.data import build_hourly_frame


def test_build_hourly_frame():
    c = [{"date": "2025-01-01T00:00:00+03:00", "consumption": 100.0}]
    g = [{
        "date": "2025-01-01T00:00:00+03:00",
        "sun": 10.0,
        "wind": 20.0,
        "river": 5.0,
        "dammedHydro": 5.0,
        "geothermal": 2.0,
        "biomass": 3.0,
        "total": 110.0,
        "naturalGas": 30.0,
        "importCoal": 20.0,
        "lignite": 15.0,
    }]
    p = [{"date": "2025-01-01T00:00:00+03:00", "price": 2000.0}]
    frame = build_hourly_frame(c, g, p)
    assert frame.iloc[0]["renewable_mwh"] == 45.0
    assert frame.iloc[0]["mcp_tl_mwh"] == 2000.0
    assert "hour_sin" in frame.columns
