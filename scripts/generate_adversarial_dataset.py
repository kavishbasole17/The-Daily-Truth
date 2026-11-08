"""
Generates a synthetic "hard negative" dataset for training-data augmentation.

Why this exists
----------------
The classical models are trained mostly on ISOT, where the *real* class is
Reuters wire copy (formal, official-sounding) and the *fake* class skews
tabloid/informal. That lets classifiers learn a shortcut: "formal tone ->
real" instead of actually reasoning about truthfulness. A well-written,
formally-worded fabricated article (e.g. from an LLM) slips right past that
shortcut.

This script generates two matched sets of formally-worded articles across
several topic domains:
  - label 0 (fake): plausible-sounding but fabricated claims, written in the
    same formal register as real wire copy (attributions, named institutions,
    no tabloid tells).
  - label 1 (real): genuinely true, uncontroversial, widely-established facts,
    also written in formal register.

Because both classes share the same *style*, the model is forced to learn
content-based signal instead of a style shortcut. Output columns match the
other dataset loaders in preprocess.py: text_content, label.

Usage:
    python scripts/generate_adversarial_dataset.py
Writes:
    dataset/Adversarial/adversarial_style.csv
"""
import itertools
import os
import random

import pandas as pd

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "dataset", "Adversarial")
OUT_PATH = os.path.join(OUT_DIR, "adversarial_style.csv")

TARGET_PER_CLASS = 150

# ── Entity pools ────────────────────────────────────────────────────────────

UNIVS = ["Stanford University", "MIT", "Cambridge University", "the University of Tokyo",
         "the University of Toronto", "ETH Zurich", "the University of Melbourne",
         "Seoul National University"]
TECH_COS = ["a major cloud computing provider", "a leading semiconductor manufacturer",
            "a top consumer electronics maker", "a widely used social media platform",
            "a prominent AI research lab", "a large telecommunications carrier"]
HEALTH_ORGS = ["the World Health Organization", "the national health ministry",
               "the Centers for Disease Control and Prevention", "a leading public health institute"]
GOV_AGENCIES = ["the national statistics office", "the transportation safety board",
                "the environmental protection agency", "the energy regulatory commission",
                "the central election commission"]
SPACE_AGENCIES = ["NASA", "the European Space Agency", "the national space research organisation"]
BANKS = ["the central bank", "a major national bank", "the reserve bank"]
COUNTRIES = ["the country", "the nation", "the region", "the province"]
PLANETS = ["Mars", "Europa", "Titan", "the far side of the Moon"]
ROVERS = ["a Mars rover", "an orbital probe", "a robotic lander"]
DISEASES = ["the common cold", "seasonal influenza", "chronic fatigue", "type 2 diabetes"]
FOODS = ["a common household spice", "an everyday citrus fruit", "a widely available root vegetable"]
JOURNALS = ["a peer-reviewed medical journal", "an internal research bulletin", "an unpublished preprint"]
VACCINES = ["the seasonal flu", "the standard childhood immunization", "the annual booster"]
NUMS = ["three", "six", "twelve", "eighteen", "twenty-four"]
YEAR_DECADES = ["the 1960s", "the 1980s", "the early 2000s", "the past three decades"]

# ── Templates: (label, template, required_slots) ────────────────────────────

TEMPLATES = [
    # Health — FAKE
    (0, ("Researchers at {univ} announced this week that {food} has been shown to reverse symptoms of "
        "{disease} within {num} days, according to a study described in {journal}.")),
    (0, ("{health_org} is reportedly reviewing an internal claim that consuming {food} daily eliminates "
        "the need for conventional treatment of {disease}, though the organisation has not confirmed the finding.")),
    (0, ("A document circulating among researchers alleges that {univ} scientists have created a single-dose "
        "treatment that fully cures {disease}, though no clinical trial data has been made public.")),
    # Health — REAL
    (1, ("{health_org} reports that global life expectancy has risen substantially since {year_decade}, "
        "driven largely by improvements in sanitation, nutrition, and access to vaccination.")),
    (1, ("{health_org} notes that decades of peer-reviewed research link regular physical activity and "
        "a balanced diet to reduced long-term cardiovascular risk.")),
    (1, ("New vaccines typically undergo years of staged clinical trials before receiving regulatory "
        "approval from health authorities such as {health_org}.")),

    # Technology — FAKE
    (0, ("According to an internal memo obtained by reporters, {tech_co} quietly disabled a chatbot "
        "after it began generating unexplained self-referential responses during testing.")),
    (0, ("Engineers at {univ} claim to have built a battery that fully recharges in under one second, "
        "though the underlying mechanism has not been disclosed or independently verified.")),
    (0, ("A leaked internal document suggests {tech_co} continues to record audio through smart-home "
        "devices even after users disable the microphone setting.")),
    # Technology — REAL
    (1, ("{tech_co} reported quarterly results that beat analyst expectations, citing continued growth "
        "in its cloud services division.")),
    (1, ("Researchers at {univ} note that advances in machine learning over {year_decade} have "
        "measurably improved the accuracy of image and speech recognition systems.")),
    (1, ("Semiconductor manufacturers have steadily increased transistor density in line with trends "
        "described by Moore's Law, though the pace of improvement has slowed in recent years.")),

    # Space — FAKE
    (0, ("Sources close to {space_agency} allege the agency is preparing to announce definitive evidence "
        "of an ancient civilization on {planet}, though officials have declined to comment.")),
    (0, ("An unnamed astronomer at {univ} claims to have detected a radio signal proving intelligent life "
        "orbits a nearby star, though the finding has not been peer reviewed.")),
    (0, ("A document circulating online alleges {space_agency} has confirmed a hidden research base on "
        "{planet}, a claim the agency has not addressed publicly.")),
    # Space — REAL
    (1, ("{space_agency}'s {rover} has spent years studying surface geology on Mars and transmitting "
        "data back to mission scientists.")),
    (1, ("Astronomers have confirmed thousands of exoplanets using data gathered by orbital telescopes "
        "over the past two decades.")),
    (1, ("{space_agency} continues to track near-Earth objects as part of an ongoing planetary defense "
        "monitoring program.")),

    # Economy — FAKE
    (0, ("An unverified memo alleges {gov_agency} is preparing to phase out physical currency within "
        "{num} months in favor of a centrally tracked digital system.")),
    (0, ("Sources close to {bank} claim the institution quietly absorbed a multi-billion-dollar trading "
        "loss that was never disclosed to regulators.")),
    (0, ("An anonymous insider alleges {country} has been issuing unreported currency to obscure the "
        "true scale of its national debt.")),
    # Economy — REAL
    (1, ("{bank} raised its benchmark interest rate in response to persistent inflation, a move aimed "
        "at moderating consumer spending.")),
    (1, ("Official statistics show {country}'s economy grew modestly last quarter, driven largely by "
        "consumer spending and exports.")),
    (1, ("Unemployment figures released by {gov_agency} showed a slight decline compared with the "
        "previous quarter.")),

    # Environment — FAKE
    (0, ("A document circulating online claims {gov_agency} has suppressed internal data showing recent "
        "climate change is part of a natural cycle unrelated to human activity.")),
    (0, ("An unverified report alleges a chemical additive under review by {gov_agency} could reverse "
        "global warming within {num} years if released into the atmosphere.")),
    (0, ("Sources allege {country} has secretly conducted large-scale cloud seeding to alter weather "
        "patterns in neighboring regions, though no evidence has been independently verified.")),
    # Environment — REAL
    (1, ("{gov_agency} affirms that scientific consensus, supported by decades of peer-reviewed "
        "research, holds that human activity is the primary driver of recent climate change.")),
    (1, "{gov_agency} data show global average temperatures have risen measurably over the past century."),
    (1, ("Renewable energy sources such as solar and wind have become increasingly cost-competitive "
        "with fossil fuels in many regions, industry reports show.")),

    # Education — FAKE
    (0, ("{univ} is said to be piloting an unverified device that allegedly uploads course material "
        "directly to students, according to a leaked internal planning document.")),
    (0, ("A document circulating among staff claims {gov_agency} plans to eliminate standardized "
        "testing nationwide next year, though no official announcement has been made.")),
    (0, ("Sources allege a private company has built a device that guarantees a top score on college "
        "entrance exams, a claim that has not been independently verified.")),
    # Education — REAL
    (1, ("{univ} reported a record number of applications for its incoming class, citing growing "
        "interest in STEM programs.")),
    (1, ("Researchers at {univ} report that early childhood education has measurable long-term "
        "benefits for academic achievement and economic outcomes.")),
    (1, ("{gov_agency} announced increased funding for public schools as part of a broader effort to "
        "address teacher shortages.")),

    # Governance — FAKE
    (0, ("A senior official who spoke on condition of anonymity alleges {country}'s government has "
        "approved a voting system that cannot be independently audited.")),
    (0, ("An unverified report claims lawmakers in {country} voted in a closed session to extend their "
        "own terms indefinitely, though no public record of such a vote exists.")),
    (0, ("Sources close to the matter allege {gov_agency} has been stockpiling emergency supplies in "
        "preparation for an unspecified event it has declined to disclose.")),
    # Governance — REAL
    (1, "Lawmakers in {country} passed a budget bill after weeks of negotiation between opposing parties."),
    (1, "{gov_agency} released its annual report outlining spending priorities for the coming fiscal year."),
    (1, "Election officials in {country} reported voter turnout consistent with historical averages."),

    # Additional REAL templates (broaden combinatorial supply so the real
    # class isn't under-represented relative to the fake templates above)
    (1, ("{univ} researchers note that further study is often needed before broad conclusions can be "
        "drawn, a standard caveat in peer-reviewed science.")),
    (1, "{gov_agency} publishes key indicators on a regular basis to inform policymakers and the public."),
    (1, ("{tech_co} has invested heavily in data center infrastructure to meet rising demand for "
        "cloud services.")),
    (1, ("{health_org} recommends routine screening as an effective way to detect conditions such as "
        "{disease} at an earlier, more treatable stage.")),
    (1, "{country} has expanded access to public transportation in major urban areas over {year_decade}."),
    (1, ("{space_agency} collaborates with international partners on missions to expand scientific "
        "understanding of the solar system.")),
    (1, ("Independent audits of {bank} are conducted periodically to verify compliance with financial "
        "regulations.")),
    (1, ("{gov_agency} publishes public safety guidance based on data collected from certified "
        "inspections.")),

    # Additional FAKE template
    (0, ("An unverified claim alleges {gov_agency} manipulated official statistics to understate "
        "{disease} case counts nationwide, though independent reviewers have not corroborated the "
        "allegation.")),
]

POOLS = {
    "univ": UNIVS, "tech_co": TECH_COS, "health_org": HEALTH_ORGS,
    "gov_agency": GOV_AGENCIES, "space_agency": SPACE_AGENCIES, "bank": BANKS,
    "country": COUNTRIES, "planet": PLANETS, "rover": ROVERS, "disease": DISEASES,
    "food": FOODS, "journal": JOURNALS, "vaccine": VACCINES, "num": NUMS,
    "year_decade": YEAR_DECADES,
}


def _slots_in(template: str) -> list[str]:
    return [name for _, name, _, _ in __import__("string").Formatter().parse(template) if name]


def generate() -> pd.DataFrame:
    rows = []
    for label, template in TEMPLATES:
        slots = _slots_in(template)
        pools = [POOLS[s] for s in slots]
        combos = list(itertools.product(*pools)) if pools else [()]
        random.shuffle(combos)
        seen = set()
        for combo in combos[:10]:  # cap per-template variants
            text = template.format(**dict(zip(slots, combo)))
            if text in seen:
                continue
            seen.add(text)
            text = text[0].upper() + text[1:]
            rows.append({"text_content": text, "label": label})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text_content"]).reset_index(drop=True)

    balanced = []
    for label in (0, 1):
        sub = df[df.label == label]
        n = min(TARGET_PER_CLASS, len(sub))
        balanced.append(sub.sample(n=n, random_state=42))
    out = pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows ({(df.label == 0).sum()} fake / {(df.label == 1).sum()} real) -> {OUT_PATH}")
