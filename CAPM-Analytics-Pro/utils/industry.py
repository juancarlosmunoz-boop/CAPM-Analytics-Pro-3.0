from __future__ import annotations

# Broad hints: Yahoo Finance taxonomy and Damodaran taxonomy are not one-to-one.
SECTOR_HINTS: dict[str, tuple[str, ...]] = {
    "Technology": ("Computer Software", "Semiconductor", "Semiconductors", "Software", "Computer Services"),
    "Financial Services": ("Banks", "Financial Services", "Investment Co.", "Insurance", "Brokerage"),
    "Healthcare": ("Drugs (Pharmaceutical)", "Healthcare Products", "Healthcare Services", "Medical Supplies"),
    "Consumer Cyclical": ("Retail (General)", "Auto & Truck", "Recreation", "Apparel", "Restaurant"),
    "Communication Services": ("Advertising", "Cable TV", "Telecom", "Entertainment", "Broadcasting"),
    "Industrials": ("Engineering/Construction", "Machinery", "Electrical Equipment", "Aerospace/Defense"),
    "Energy": ("Oilfield Services/Equipment", "Oil/Gas (Production and Exploration)", "Oil/Gas (Integrated)", "Utility (Foreign)"),
    "Utilities": ("Utility (General)", "Utility (Water)", "Power", "Natural Gas"),
    "Basic Materials": ("Metals & Mining", "Chemical (Basic)", "Steel", "Chemical (Specialty)"),
    "Real Estate": ("R.E.I.T.", "Real Estate (Operations & Services)"),
    "Consumer Defensive": ("Beverage", "Food Processing", "Retail (Grocery and Food)"),
}

def suggested_industries(sector: str, damodaran_industries: list[str], industry: str = "") -> list[str]:
    """Rank plausible Damodaran industries using Yahoo sector/industry hints."""
    ranked: list[tuple[int, str]] = []
    sector_hints = SECTOR_HINTS.get(sector, ())
    for name in damodaran_industries:
        low = name.lower()
        score = 0
        if industry and industry.lower() in low:
            score += 12
        if any(h.lower() in low for h in sector_hints):
            score += 5
        if score:
            ranked.append((score, name))
    ranked.sort(key=lambda item: (-item[0], item[1].lower()))
    return [name for _, name in ranked]
