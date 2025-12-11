import sqlite3
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === CONFIG ===
DB_PATH = "C:\\Users\\henry\\Data Projects\\Stock_Optimization\\finviz_news.db"  # <-- change if your DB filename is different
TABLE_NAME = "finviz_news_highfreq_table"  # <-- change if you used another table name
ID_COL = "id"
TITLE_COL = "title"
EXPANDED_COL = "expanded"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # free, small-ish instruction model
MAX_NEW_TOKENS = 80   # ~1–3 sentences
BATCH_COMMIT = 50     # commit every N updates


# === LOAD MODEL ===
print("Loading model (first time may take a bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",        # GPU if available, otherwise CPU
)

PROMPT_TEMPLATE = """You are a financial news explainer.

Expand the following stock market news headline into 2 concise sentences that
explain what the news likely means for investors. Do NOT invent specific numbers
or events that are not implied; just explain the implications in general terms.

Headline: "{headline}"
Expanded explanation:"""


def expand_headline(headline: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate a short 1–3 sentence explanation for a headline."""
    safe_headline = headline.replace('"', '\\"')
    prompt = PROMPT_TEMPLATE.format(headline=safe_headline)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # deterministic, more stable
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Prefer the part after our marker
    if "Expanded explanation:" in full_text:
        expanded = full_text.split("Expanded explanation:", 1)[1].strip()
    else:
        # Fallback: strip off the prompt
        expanded = full_text[len(prompt):].strip()

    # Safety: trim anything excessively long
    return expanded[:1000]


def ensure_expanded_column(cur: sqlite3.Cursor):
    """Add EXPANDED_COL to TABLE_NAME if it doesn't exist yet."""
    cur.execute(f"PRAGMA table_info({TABLE_NAME});")
    cols = [row[1] for row in cur.fetchall()]
    if EXPANDED_COL not in cols:
        print(f"Adding '{EXPANDED_COL}' column to {TABLE_NAME}...")
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {EXPANDED_COL} TEXT;")


def main():
    # === CONNECT TO DB ===
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ensure expanded column exists
    ensure_expanded_column(cur)
    conn.commit()

    # Fetch all rows in the new table that still need expansions
    query = f"""
        SELECT {ID_COL}, {TITLE_COL}
        FROM {TABLE_NAME}
        WHERE {EXPANDED_COL} IS NULL
    """
    cur.execute(query)
    rows = cur.fetchall()
    print(f"Found {len(rows)} headlines in {TABLE_NAME} without expansions.")

    updated = 0

    for row_id, title in tqdm(rows, desc="Expanding headlines"):
        if not title:
            expanded = ""
        else:
            try:
                expanded = expand_headline(title)
            except Exception as e:
                print(f"\nError expanding id={row_id}, title={title!r}: {e}")
                expanded = ""

        cur.execute(
            f"UPDATE {TABLE_NAME} SET {EXPANDED_COL} = ? WHERE {ID_COL} = ?;",
            (expanded, row_id),
        )

        updated += 1
        if updated % BATCH_COMMIT == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print(f"Done. Expanded {updated} headlines in {TABLE_NAME}.")


if __name__ == "__main__":
    main()
