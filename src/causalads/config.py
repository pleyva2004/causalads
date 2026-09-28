"""Paths, seeds, and business assumptions shared by every stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SEED = 42
# Seeds used wherever a result depends on randomness we want to average over.
SEEDS = (0, 1, 2, 3, 4)
N_BOOTSTRAP = 2000

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
GEOLIFT_URLS = {
    "GeoLift_PreTest.rda": "https://raw.githubusercontent.com/facebookincubator/GeoLift/main/data/GeoLift_PreTest.rda",
    "GeoLift_Test.rda": "https://raw.githubusercontent.com/facebookincubator/GeoLift/main/data/GeoLift_Test.rda",
}

# Split fractions (stratified on arm x visit x conversion).
TEST_FRAC = 0.20
VAL_FRAC = 0.20

# Hillstrom randomized each customer to one of three arms with equal probability.
ARMS = {0: "No E-Mail", 1: "Mens E-Mail", 2: "Womens E-Mail"}
P_ARM = 1.0 / 3.0
P_ANY_EMAIL = 2.0 / 3.0

# Business assumption: the dataset has no cost column. This is ILLUSTRATIVE and
# only used for iROAS; break-even cost per contact is reported alongside it so no
# conclusion depends on this number.
COST_PER_CONTACT = 0.10  # USD per email sent

# Fraction of the customer base the fixed budget can reach.
BUDGET_FRACS = (0.10, 0.20, 0.30, 0.50)
