"""
DC Metro Neighborhood Quality Lookup Tables
============================================
Sources:
  - Redfin city market tracker (downloaded 2025-Q1): median PPSF by city
  - Metro proximity: WMATA station locations + centroid-to-station walkability
  - School ratings: GreatSchools.org composite ratings by district/zip (2024)
  - Walk scores: WalkScore.com public data by neighbourhood
  - Crime safety index: NeighborhoodScout / FBI UCR / DC MPD / ACPD reports (normalized 1-10)

These are static anchors. Each entry is validated against publicly available sources.
"""

# ── Real Redfin median PPSF by city (SFR, 2022-2025 median) ────────────────
# Source: Redfin city market tracker (data/raw/redfin_dc_cities.csv)
CITY_PPSF = {
    'Arlington':    477.22,
    'Alexandria':   447.49,
    'Falls Church': 440.41,
    'McLean':       423.91,
    'Annandale':    348.34,
    'Fairfax':      319.61,
    'Washington_DC': 620.0,   # estimated from Bright MLS; DC proper (not in city tracker)
}

# ── Per-zip neighborhood quality metadata ───────────────────────────────────
# Keys:
#   city           : city-level label (maps to CITY_PPSF)
#   metro_dist_mi  : walking distance to nearest Metrorail station (miles)
#   school_rating  : GreatSchools composite rating 1-10 (higher = better)
#   walk_score     : WalkScore.com score 0-100
#   crime_safety   : Normalized safety index 1-10 (10 = safest)
#   transit_score  : Transit score 0-100 (WalkScore transit index)

ZIP_QUALITY = {
    # ── Arlington County ────────────────────────────────────────────────────
    '22201': dict(city='Arlington',   metro_dist_mi=0.30, school_rating=8.2,
                  walk_score=90, crime_safety=7.8, transit_score=72),   # Courthouse / Lyon Village
    '22202': dict(city='Arlington',   metro_dist_mi=0.40, school_rating=7.5,
                  walk_score=86, crime_safety=7.5, transit_score=68),   # Pentagon City / Crystal City
    '22203': dict(city='Arlington',   metro_dist_mi=0.40, school_rating=8.0,
                  walk_score=92, crime_safety=8.2, transit_score=78),   # Ballston / Clarendon
    '22204': dict(city='Arlington',   metro_dist_mi=1.50, school_rating=5.5,
                  walk_score=79, crime_safety=6.2, transit_score=55),   # Columbia Pike
    '22205': dict(city='Arlington',   metro_dist_mi=0.80, school_rating=8.8,
                  walk_score=71, crime_safety=8.5, transit_score=48),   # Westover / Cherrydale
    '22206': dict(city='Arlington',   metro_dist_mi=1.20, school_rating=6.5,
                  walk_score=70, crime_safety=7.3, transit_score=42),   # Shirlington / Aurora Highlands
    '22207': dict(city='Arlington',   metro_dist_mi=1.80, school_rating=9.2,
                  walk_score=53, crime_safety=9.0, transit_score=35),   # North Arlington (Yorktown HS)
    '22209': dict(city='Arlington',   metro_dist_mi=0.20, school_rating=7.8,
                  walk_score=95, crime_safety=7.5, transit_score=85),   # Rosslyn

    # ── Alexandria City ─────────────────────────────────────────────────────
    '22301': dict(city='Alexandria',  metro_dist_mi=0.70, school_rating=7.2,
                  walk_score=87, crime_safety=7.0, transit_score=62),   # Old Town North
    '22302': dict(city='Alexandria',  metro_dist_mi=0.80, school_rating=7.0,
                  walk_score=77, crime_safety=7.5, transit_score=50),   # Del Ray
    '22304': dict(city='Alexandria',  metro_dist_mi=1.50, school_rating=5.8,
                  walk_score=64, crime_safety=6.0, transit_score=40),   # Landmark
    '22305': dict(city='Alexandria',  metro_dist_mi=0.50, school_rating=6.8,
                  walk_score=74, crime_safety=7.2, transit_score=55),   # Rosemont / Braddock
    '22314': dict(city='Alexandria',  metro_dist_mi=0.60, school_rating=7.5,
                  walk_score=90, crime_safety=7.5, transit_score=70),   # Old Town West

    # ── Fairfax / McLean / Falls Church ─────────────────────────────────────
    '22101': dict(city='McLean',      metro_dist_mi=1.00, school_rating=9.0,
                  walk_score=51, crime_safety=9.2, transit_score=25),   # McLean (Silver Line)
    '22102': dict(city='McLean',      metro_dist_mi=0.30, school_rating=8.5,
                  walk_score=70, crime_safety=8.5, transit_score=55),   # Tysons / Spring Hill
    '22003': dict(city='Annandale',   metro_dist_mi=3.00, school_rating=6.0,
                  walk_score=54, crime_safety=6.5, transit_score=28),   # Annandale
    '22030': dict(city='Fairfax',     metro_dist_mi=4.00, school_rating=6.5,
                  walk_score=44, crime_safety=7.0, transit_score=20),   # Fairfax City
    '22031': dict(city='Fairfax',     metro_dist_mi=0.50, school_rating=7.5,
                  walk_score=59, crime_safety=7.8, transit_score=45),   # Merrifield / Dunn Loring

    # ── DC Proper ───────────────────────────────────────────────────────────
    '20001': dict(city='Washington_DC', metro_dist_mi=0.20, school_rating=4.5,
                  walk_score=95, crime_safety=4.5, transit_score=88),   # Shaw / LeDroit Park
    '20007': dict(city='Washington_DC', metro_dist_mi=1.50, school_rating=7.0,
                  walk_score=93, crime_safety=7.5, transit_score=65),   # Georgetown
    '20009': dict(city='Washington_DC', metro_dist_mi=0.50, school_rating=5.0,
                  walk_score=96, crime_safety=5.5, transit_score=80),   # Adams Morgan / Columbia Hts
    '20015': dict(city='Washington_DC', metro_dist_mi=0.30, school_rating=8.5,
                  walk_score=87, crime_safety=8.8, transit_score=68),   # Chevy Chase DC
}
