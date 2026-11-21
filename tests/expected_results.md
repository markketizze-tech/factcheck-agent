# Sample trap document — expected ground-truth checks

This is a **synthetic test fixture**, not the user's missing reference PDF. It is designed to contain one stale assertion, one deliberately false figure, and two supportable facts. Live verdicts may be `INSUFFICIENT EVIDENCE` if the search provider does not retrieve enough evidence; the app must not force an expected verdict without sources.

| Claim in sample PDF | Expected assessment when adequate evidence is retrieved | Ground-truth source |
|---|---|---|
| Pluto is currently the ninth major planet in the Solar System. | INACCURATE (outdated classification; it was reclassified as a dwarf planet in 2006) | NASA: https://science.nasa.gov/solar-system/planets/ and https://science.nasa.gov/dwarf-planets/pluto/facts/ |
| Earth's equatorial circumference is 50,000 km. | FALSE (NASA sources place it at about 40,075 km / 40,030 km depending on reference value used) | NASA: https://www.nasa.gov/space-science-and-astrobiology-at-ames/interesting-fact-of-the-month-current/interesting-fact-of-the-month-2020/ and https://science.nasa.gov/moon/by-the-numbers/ |
| The average Earth–Moon distance is approximately 384,400 km. | VERIFIED | NASA: https://science.nasa.gov/moon/by-the-numbers/ |
| Earth is about 150 million km from the Sun on average. | VERIFIED | NASA: https://science.nasa.gov/earth/facts/ |

The circumference reference deliberately demonstrates source/definition nuance: NASA pages show both roughly 40,075 km and 40,030.2 km, depending on the source's cited radius value. The app should identify that both are far from 50,000 km, while noting the scope/definition rather than inventing false precision.
