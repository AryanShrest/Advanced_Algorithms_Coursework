"""
nepal_cities.py

A list of real Nepali cities/towns (a mix of major metros, regional hubs,
and district headquarters spread across all seven provinces) used to give
realistic, recognisable names to the City records in Task 1, instead of
generic placeholders like "City0", "City1", etc.

get_city_name(index) cycles through the list and appends a numbered
suffix once the index exceeds the number of real names available -- this
is only needed for the larger empirical benchmark datasets (n=1,000 and
n=10,000), where we need far more records than there are distinct towns
in Nepal. Small demo datasets (7-9 cities) never hit the suffix branch,
so they show clean, real names like "Kathmandu", "Dharan", "Pokhara".
"""

NEPAL_CITIES = [
    "Kathmandu", "Pokhara", "Lalitpur", "Bhaktapur", "Biratnagar",
    "Birgunj", "Dharan", "Butwal", "Dhangadhi", "Nepalgunj",
    "Itahari", "Hetauda", "Janakpur", "Bharatpur", "Damak",
    "Tikapur", "Gorkha", "Ilam", "Tansen", "Baglung",
    "Rajbiraj", "Siddharthanagar", "Dhankuta", "Taplejung", "Salyan",
    "Jumla", "Jomsom", "Sindhuli", "Charikot", "Ramechhap",
    "Okhaldhunga", "Bhojpur", "Terhathum", "Phidim", "Khandbari",
    "Salleri", "Diktel", "Triyuga", "Rajbiraj", "Lahan",
    "Malangwa", "Jaleshwar", "Gaur", "Kalaiya", "Birendranagar",
    "Dailekh", "Jajarkot", "Dunai", "Kalikot", "Simikot",
    "Bajura", "Chainpur", "Sanfebagar", "Dipayal", "Kanchanpur",
    "Dadeldhura", "Baitadi", "Darchula", "Damauli", "Besisahar",
    "Waling", "Bandipur", "Beni", "Kusma", "Ridi",
    "Bhairahawa", "Taulihawa", "Ghorahi", "Tulsipur", "Musikot",
    "Rukumkot", "Manma", "Inaruwa", "Mechinagar", "Kakarvitta",
]


def get_city_name(index: int) -> str:
    """Return a real Nepal place name for the given index. Cycles through
    the list with a numeric suffix (e.g. 'Kathmandu (2)') once index
    exceeds the number of distinct names available."""
    n = len(NEPAL_CITIES)
    base_name = NEPAL_CITIES[index % n]
    cycle = index // n
    if cycle == 0:
        return base_name
    return f"{base_name} ({cycle + 1})"
