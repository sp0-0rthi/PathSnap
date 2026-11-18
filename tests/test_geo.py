"""Sanity tests for the geodesy helpers."""
from src.geo import haversine_m


def test_haversine_one_degree_equator():
    # one degree of longitude at the equator is ~111.19 km
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert abs(d - 111_195) < 500


def test_haversine_zero_for_identical_points():
    assert haversine_m(12.9716, 77.5946, 12.9716, 77.5946) == 0.0


def test_haversine_latitude_degree_shorter_at_high_lat():
    # a longitude degree shrinks with latitude
    equator = haversine_m(0.0, 0.0, 0.0, 1.0)
    oslo = haversine_m(60.0, 0.0, 60.0, 1.0)
    assert oslo < equator * 0.55
