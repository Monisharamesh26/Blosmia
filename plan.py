import os
import datetime
import sqlite3
from database import dbfunctions2 as db  # dbfunctions2 points to database_wbc.db

# =========================
# CONFIG
# =========================
BASE_DIR = r"E:\Project25\WBC_DATA"

# Map: cell key -> (smear_folder, cropped_folder, CellsName for DB lookup)
# CellsName is looked up case-insensitively from CellsType to get CellsId safely.
FOLDER_MAP = {
    "Basophil": ("basophil", "BASOcropped", "Basophil"),
    "Neutrophil": ("neutrophil", "NEUcropped", "Neutrophil"),
    "Eosinophil": ("eosinophil", "EOScropped", "Eosinophil"),
    "Lymphocyte": ("lymphocyte", "LYMcropped", "Lymphocyte"),
    "Monocyte": ("monocyte", "MONOcropped", "Monocyte"),
}

# <<< SET YOUR THREE PATIENT IDS HERE >>>
PATIENT_IDS = [101, 102, 103]

# Extensions to accept for images
IMG_EXTS = (".jpg", ".jpeg", ".png")


# =========================
# HELPERS
# =========================
def db_path():
    """Try to use db.DATABASE_PATH; fall back to the known path if needed."""
    path = getattr(db, "DATABASE_PATH", None)
    if path:
        return path
    # Fallback (keep in sync with dbfunctions2)
    return r"E:\Project25\J_Blosmia1\database\database_wbc.db"


def resolve_cells_id(cells_name: str) -> int:
    """
    Resolve CellsId by CellsName from CellsType, case-insensitive.
    Raises RuntimeError if not found.
    """
    con = sqlite3.connect(db_path())
    cur = con.cursor()
    cur.execute(
        "SELECT CellsId FROM CellsType WHERE LOWER(CellsName)=LOWER(?)",
        (cells_name,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        raise RuntimeError(f"CellsName '{cells_name}' not found in CellsType table.")
    return int(row[0])


def contiguous_3way_split(items):
    """
    Contiguously split a list into 3 slices for the 3 patients:
    First slice gets the first extra if remainder exists (e.g., 76 -> 26,25,25).
    Returns a list of 3 sublists.
    """
    n = len(items)
    base = n // 3
    rem = n % 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    slices = []
    start = 0
    for sz in sizes:
        end = start + sz
        slices.append(items[start:end])
        start = end
    return slices


def chunk_by_3(items):
    """Yield contiguous chunks of size 3 (last chunk may be 1–2)."""
    for i in range(0, len(items), 3):
        yield items[i:i+3]


def build_blob_index(blob_dir):
    """
    Build an index: smear_base -> [full_blob_paths...]
    Blob files are 'crop_{smear_base}_{index}.ext'. We strip the last '_{index}'.
    """
    index = {}
    if not os.path.exists(blob_dir):
        return index
    for fname in os.listdir(blob_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        base, _ext = os.path.splitext(fname)
        if not base.startswith("crop_"):
            continue
        # Remove trailing '_<index>' safely (smear_base may contain underscores)
        after_crop = base[5:]  # strip 'crop_'
        smear_key = after_crop.rsplit("_", 1)[0]  # remove only the last _idx
        index.setdefault(smear_key, []).append(os.path.join(blob_dir, fname))
    return index


def print_report_summary(rep_id):
    """Print counts for a report from ReportResult (or live counts if needed)."""
    con = sqlite3.connect(db_path())
    cur = con.cursor()
    # Try ReportResult first
    cur.execute(
        """SELECT basophil, eosinophil, Lymphocyte, Monocyte, Neutrophil
           FROM ReportResult WHERE report_id=?""",
        (rep_id,),
    )
    row = cur.fetchone()
    con.close()

    if row:
        b, e, l, m, n = row
        print(f"    Basophil={b}, Eosinophil={e}, Lymphocyte={l}, Monocyte={m}, Neutrophil={n}")
    else:
        # Fallback direct count (rare, if not inserted)
        types_a = db.get_total_count(rep_id)
        counts = [c[0] for c in types_a]
        # We print in the canonical WBC order defined above if lengths match
        labels = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
        joined = ", ".join(f"{labels[i]}={counts[i]}" for i in range(min(len(counts), 5)))
        print(f"    {joined}")


# =========================
# MAIN MIGRATION LOGIC
# =========================
def migrate():
    # Resolve CellsIds once
    cell_id_map = {}
    for key, (_smear, _crop, cells_name) in FOLDER_MAP.items():
        cell_id_map[key] = resolve_cells_id(cells_name)

    created_reports = []  # [(patient_id, cell_key, rep_id, num_smears, num_blobs), ...]

    for cell_key, (smear_folder, crop_folder, _cells_name) in FOLDER_MAP.items():
        smear_dir = os.path.join(BASE_DIR, smear_folder)
        blob_dir = os.path.join(BASE_DIR, crop_folder)

        if not os.path.exists(smear_dir):
            print(f"⚠ Smear folder missing for {cell_key}: {smear_dir} — skipping this type.")
            continue
        if not os.path.exists(blob_dir):
            print(f"⚠ Cropped folder missing for {cell_key}: {blob_dir} — skipping this type.")
            continue

        # Collect & sort smear files (deterministic order)
        smear_files = sorted([f for f in os.listdir(smear_dir) if f.lower().endswith(IMG_EXTS)])

        # Build blob index once (smear_base -> list of blob paths)
        blob_index = build_blob_index(blob_dir)

        # Contiguously split smears for the three patients
        p_slices = contiguous_3way_split(smear_files)

        for p_idx, patient_id in enumerate(PATIENT_IDS):
            patient_slice = p_slices[p_idx]
            if not patient_slice:
                continue

            # Create reports in batches of 3 smears
            for batch in chunk_by_3(patient_slice):
                # One report per batch of 3 smears
                rep_id = db.report([patient_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

                batch_blob_count = 0
                for smear_file in batch:
                    smear_path = os.path.join(smear_dir, smear_file)
                    with open(smear_path, "rb") as f:
                        img_blob = f.read()
                    # Insert smear -> image_data
                    img_id = db.addimage(rep_id, smear_file, img_blob, smear_dir)

                    smear_base, _ = os.path.splitext(smear_file)
                    # Get all blobs for this smear
                    for blob_path in blob_index.get(smear_base, []):
                        with open(blob_path, "rb") as bf:
                            blob_img = bf.read()
                        db.types(img_id, rep_id, blob_img, cell_id_map[cell_key])
                        batch_blob_count += 1

                # Insert counts for this report into ReportResult
                types_a = db.get_total_count(rep_id)
                cell_count_data = [rep_id] + [count[0] for count in types_a]
                db.add_count_in_reportResult(cell_count_data)

                created_reports.append((patient_id, cell_key, rep_id, len(batch), batch_blob_count))
                print(f"➡ Created report {rep_id} for patient {patient_id} [{cell_key}] "
                      f"with {len(batch)} smear(s), {batch_blob_count} blob(s).")
                print_report_summary(rep_id)

    # Final recap
    print("\n================ ALL REPORTS CREATED ================\n")
    for patient_id, cell_key, rep_id, n_smears, n_blobs in created_reports:
        print(f"Patient {patient_id} | {cell_key:<11} | Report {rep_id} | smears={n_smears} | blobs={n_blobs}")
    print("\nDone.")


if __name__ == "__main__":
    migrate()
