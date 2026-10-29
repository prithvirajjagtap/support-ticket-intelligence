from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "support_tickets_raw.csv"
CLEAN_PATH = BASE_DIR / "data" / "tickets_clean.csv"

df = pd.read_csv(RAW_PATH)

# Keep English tickets only.
df = df[df["language"].astype(str).str.lower().eq("en")].copy()

# Build the text from the subject and body.
subject = df["subject"].fillna("").astype(str).str.strip()
body = df["body"].fillna("").astype(str).str.strip()

df["ticket_text"] = (
    "Subject: " + subject + "\n" + "Message: " + body
).str.strip()

# Remove rows without usable text or labels.
df = df[df["ticket_text"].str.len() >= 10]
df = df.dropna(subset=["queue", "priority"])

# Normalize labels.
df["category"] = df["queue"].astype(str).str.strip()
df["priority"] = df["priority"].astype(str).str.strip().str.title()

df = df[df["category"].ne("") & df["priority"].isin(["Low", "Medium", "High"])]

# Remove exact duplicate ticket text.
df = df.drop_duplicates(subset=["ticket_text"]).reset_index(drop=True)

# Save only the columns needed for model training.
clean = df[["ticket_text", "category", "priority"]]
clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8")

print("Dataset preparation complete.")
print("Clean tickets:", len(clean))
print("Categories:", clean["category"].nunique())
print("\nCategory counts:")
print(clean["category"].value_counts().to_string())
print("\nPriority counts:")
print(clean["priority"].value_counts().to_string())
print("\nSaved to:", CLEAN_PATH)
