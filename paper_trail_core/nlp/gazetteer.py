"""
Small built-in gazetteer: countries, Indian states / union territories, and
well-known cities.

It does two jobs that matter for the locator and neither needs the network:

  * **Cheap, certain hits.** "Maharashtra" or "India" never needs a geocoder
    round trip, which keeps the locator fast and usable offline.
  * **Context for disambiguation.** Knowing an article mentions "Kerala" is
    what tells the geocoder that its "Kochi" is the Indian one.

Deliberately small. It is not meant to be exhaustive — anything it does not
know goes to the geocoder. It is meant to be *right*, so ambiguous names are
either omitted or flagged and never silently resolved to the wrong country.
"""
from __future__ import annotations

from typing import NamedTuple


class GazEntry(NamedTuple):
    name: str                 # canonical display name
    level: str                # country | state | city
    country_code: str         # ISO 3166-1 alpha-2, lowercase
    country: str
    state: str | None = None


# ── Countries ─────────────────────────────────────────────────────────────────

_COUNTRIES: dict[str, tuple[str, str]] = {
    # name -> (canonical, iso2)
    "india": ("India", "in"), "pakistan": ("Pakistan", "pk"),
    "bangladesh": ("Bangladesh", "bd"), "sri lanka": ("Sri Lanka", "lk"),
    "nepal": ("Nepal", "np"), "bhutan": ("Bhutan", "bt"),
    "china": ("China", "cn"), "japan": ("Japan", "jp"),
    "south korea": ("South Korea", "kr"), "north korea": ("North Korea", "kp"),
    "indonesia": ("Indonesia", "id"), "malaysia": ("Malaysia", "my"),
    "singapore": ("Singapore", "sg"), "thailand": ("Thailand", "th"),
    "vietnam": ("Vietnam", "vn"), "philippines": ("Philippines", "ph"),
    "afghanistan": ("Afghanistan", "af"), "iran": ("Iran", "ir"),
    "iraq": ("Iraq", "iq"), "israel": ("Israel", "il"),
    "saudi arabia": ("Saudi Arabia", "sa"), "united arab emirates": ("United Arab Emirates", "ae"),
    "uae": ("United Arab Emirates", "ae"), "qatar": ("Qatar", "qa"),
    "turkey": ("Turkey", "tr"), "egypt": ("Egypt", "eg"),
    "nigeria": ("Nigeria", "ng"), "kenya": ("Kenya", "ke"),
    "south africa": ("South Africa", "za"), "ethiopia": ("Ethiopia", "et"),
    "ghana": ("Ghana", "gh"),
    "united states": ("United States", "us"), "united states of america": ("United States", "us"),
    "usa": ("United States", "us"), "u.s.": ("United States", "us"),
    "canada": ("Canada", "ca"), "mexico": ("Mexico", "mx"),
    "brazil": ("Brazil", "br"), "argentina": ("Argentina", "ar"),
    "united kingdom": ("United Kingdom", "gb"), "uk": ("United Kingdom", "gb"),
    "england": ("England", "gb"), "scotland": ("Scotland", "gb"), "wales": ("Wales", "gb"),
    "ireland": ("Ireland", "ie"), "france": ("France", "fr"),
    "germany": ("Germany", "de"), "italy": ("Italy", "it"),
    "spain": ("Spain", "es"), "portugal": ("Portugal", "pt"),
    "netherlands": ("Netherlands", "nl"), "belgium": ("Belgium", "be"),
    "switzerland": ("Switzerland", "ch"), "sweden": ("Sweden", "se"),
    "norway": ("Norway", "no"), "denmark": ("Denmark", "dk"),
    "finland": ("Finland", "fi"), "poland": ("Poland", "pl"),
    "ukraine": ("Ukraine", "ua"), "russia": ("Russia", "ru"),
    "australia": ("Australia", "au"), "new zealand": ("New Zealand", "nz"),
}

# ── Indian states and union territories ───────────────────────────────────────

_IN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh",
    "Andaman and Nicobar Islands", "Lakshadweep",
]
_IN_STATE_ALIASES = {"orissa": "Odisha", "nct of delhi": "Delhi", "new delhi": "Delhi",
                     "pondicherry": "Puducherry", "j&k": "Jammu and Kashmir",
                     "kashmir": "Jammu and Kashmir", "kashmir valley": "Jammu and Kashmir"}

# city -> state
_IN_CITIES: dict[str, str] = {
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra", "Thane": "Maharashtra", "Aurangabad": "Maharashtra",
    "Kolhapur": "Maharashtra", "Solapur": "Maharashtra",
    "Bengaluru": "Karnataka", "Bangalore": "Karnataka", "Mysuru": "Karnataka",
    "Mysore": "Karnataka", "Mangaluru": "Karnataka", "Hubballi": "Karnataka",
    "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Madurai": "Tamil Nadu",
    "Tiruchirappalli": "Tamil Nadu", "Salem": "Tamil Nadu",
    "Hyderabad": "Telangana", "Warangal": "Telangana",
    "Visakhapatnam": "Andhra Pradesh", "Vijayawada": "Andhra Pradesh",
    "Tirupati": "Andhra Pradesh", "Amaravati": "Andhra Pradesh",
    "Kochi": "Kerala", "Thiruvananthapuram": "Kerala", "Kozhikode": "Kerala",
    "Thrissur": "Kerala", "Kannur": "Kerala",
    "Kolkata": "West Bengal", "Howrah": "West Bengal", "Siliguri": "West Bengal",
    "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat",
    "Rajkot": "Gujarat", "Gandhinagar": "Gujarat",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan", "Udaipur": "Rajasthan",
    "Kota": "Rajasthan", "Ajmer": "Rajasthan", "Bikaner": "Rajasthan",
    "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh", "Varanasi": "Uttar Pradesh",
    "Agra": "Uttar Pradesh", "Noida": "Uttar Pradesh", "Ghaziabad": "Uttar Pradesh",
    "Prayagraj": "Uttar Pradesh", "Allahabad": "Uttar Pradesh", "Meerut": "Uttar Pradesh",
    "Bhopal": "Madhya Pradesh", "Indore": "Madhya Pradesh", "Gwalior": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh", "Ujjain": "Madhya Pradesh",
    "Patna": "Bihar", "Gaya": "Bihar", "Muzaffarpur": "Bihar",
    "Ranchi": "Jharkhand", "Jamshedpur": "Jharkhand", "Dhanbad": "Jharkhand",
    "Bhubaneswar": "Odisha", "Cuttack": "Odisha", "Puri": "Odisha",
    "Raipur": "Chhattisgarh", "Bilaspur": "Chhattisgarh",
    "Guwahati": "Assam", "Dibrugarh": "Assam", "Silchar": "Assam",
    "Shillong": "Meghalaya", "Imphal": "Manipur", "Aizawl": "Mizoram",
    "Kohima": "Nagaland", "Agartala": "Tripura", "Gangtok": "Sikkim",
    "Itanagar": "Arunachal Pradesh",
    "Dehradun": "Uttarakhand", "Haridwar": "Uttarakhand", "Rishikesh": "Uttarakhand",
    "Shimla": "Himachal Pradesh", "Manali": "Himachal Pradesh",
    "Amritsar": "Punjab", "Ludhiana": "Punjab", "Jalandhar": "Punjab",
    "Gurugram": "Haryana", "Gurgaon": "Haryana", "Faridabad": "Haryana",
    "Panipat": "Haryana",
    "Srinagar": "Jammu and Kashmir", "Jammu": "Jammu and Kashmir",
    "Leh": "Ladakh", "Panaji": "Goa", "Margao": "Goa",
}

# ── A few worldwide cities, mostly to supply country context ─────────────────

_WORLD_CITIES: dict[str, tuple[str, str, str]] = {
    # name -> (canonical, iso2, country)
    "london": ("London", "gb", "United Kingdom"), "manchester": ("Manchester", "gb", "United Kingdom"),
    "paris": ("Paris", "fr", "France"), "berlin": ("Berlin", "de", "Germany"),
    "madrid": ("Madrid", "es", "Spain"), "rome": ("Rome", "it", "Italy"),
    "moscow": ("Moscow", "ru", "Russia"), "kyiv": ("Kyiv", "ua", "Ukraine"),
    "beijing": ("Beijing", "cn", "China"), "shanghai": ("Shanghai", "cn", "China"),
    "tokyo": ("Tokyo", "jp", "Japan"), "seoul": ("Seoul", "kr", "South Korea"),
    "bangkok": ("Bangkok", "th", "Thailand"), "jakarta": ("Jakarta", "id", "Indonesia"),
    "dubai": ("Dubai", "ae", "United Arab Emirates"), "abu dhabi": ("Abu Dhabi", "ae", "United Arab Emirates"),
    "riyadh": ("Riyadh", "sa", "Saudi Arabia"), "tehran": ("Tehran", "ir", "Iran"),
    "kabul": ("Kabul", "af", "Afghanistan"), "islamabad": ("Islamabad", "pk", "Pakistan"),
    "karachi": ("Karachi", "pk", "Pakistan"), "lahore": ("Lahore", "pk", "Pakistan"),
    "dhaka": ("Dhaka", "bd", "Bangladesh"), "colombo": ("Colombo", "lk", "Sri Lanka"),
    "kathmandu": ("Kathmandu", "np", "Nepal"), "cairo": ("Cairo", "eg", "Egypt"),
    "lagos": ("Lagos", "ng", "Nigeria"), "nairobi": ("Nairobi", "ke", "Kenya"),
    "johannesburg": ("Johannesburg", "za", "South Africa"),
    "new york": ("New York", "us", "United States"), "los angeles": ("Los Angeles", "us", "United States"),
    "chicago": ("Chicago", "us", "United States"), "houston": ("Houston", "us", "United States"),
    "san francisco": ("San Francisco", "us", "United States"),
    "toronto": ("Toronto", "ca", "Canada"), "vancouver": ("Vancouver", "ca", "Canada"),
    "sydney": ("Sydney", "au", "Australia"), "melbourne": ("Melbourne", "au", "Australia"),
    "auckland": ("Auckland", "nz", "New Zealand"), "singapore city": ("Singapore", "sg", "Singapore"),
    "sao paulo": ("São Paulo", "br", "Brazil"), "mexico city": ("Mexico City", "mx", "Mexico"),
}

# Gazetteer names that are also ordinary words or exist in several countries.
# They still resolve, but only when the text gives a place cue ("in Salem").
_AMBIGUOUS = {"salem", "kota", "gaya", "puri"}

_BY_NAME: dict[str, GazEntry] = {}


def _build() -> None:
    for key, (name, cc) in _COUNTRIES.items():
        _BY_NAME[key] = GazEntry(name, "country", cc, name)
    for s in _IN_STATES:
        _BY_NAME[s.lower()] = GazEntry(s, "state", "in", "India", s)
    for alias, canon in _IN_STATE_ALIASES.items():
        _BY_NAME[alias] = GazEntry(canon, "state", "in", "India", canon)
    for city, state in _IN_CITIES.items():
        _BY_NAME[city.lower()] = GazEntry(city, "city", "in", "India", state)
    for key, (name, cc, country) in _WORLD_CITIES.items():
        _BY_NAME.setdefault(key, GazEntry(name, "city", cc, country))


_build()

# Words that begin a capitalised run but are not part of a place name.
NOT_PLACE_WORDS = frozenset(["the", "a", "an", "this", "that", "these", "those", "his", "her", "their", "our", "your", "its", "mr", "mrs", "ms", "dr", "shri", "smt", "prof", "sir", "madam", "prime", "minister", "president", "chief", "secretary", "governor", "chancellor", "mayor", "union", "state", "central", "national", "federal", "government", "ministry", "department", "police", "court", "supreme", "high", "district", "sessions", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "according", "however", "meanwhile", "also", "but", "and", "or", "if", "when", "while", "after", "before", "officials", "sources", "scientists", "researchers", "experts", "residents", "authorities"])


def lookup(name: str) -> GazEntry | None:
    """Exact (case-insensitive) gazetteer entry for a place name, or None."""
    return _BY_NAME.get(" ".join(name.lower().split()))


def is_ambiguous(name: str) -> bool:
    return name.lower() in _AMBIGUOUS


def all_names() -> list[str]:
    return list(_BY_NAME)
