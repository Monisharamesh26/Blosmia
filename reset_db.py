import sqlite3

DB_PATH = r"E:\Project25\J_Blosmia1\database\database_wbc.db"

def reset():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Step 1: Delete data (order matters due to foreign keys)
    cur.execute("DELETE FROM ReportResult")
    cur.execute("DELETE FROM BLOB_TAB")
    cur.execute("DELETE FROM image_data")
    cur.execute("DELETE FROM report")
    # ⚠ Do NOT clear patient_details unless you want to reinsert patients

    # Step 2: Reset autoincrement counters
    cur.execute("DELETE FROM sqlite_sequence WHERE name='report'")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='image_data'")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='BLOB_TAB'")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='ReportResult'")

    con.commit()
    con.close()
    print("✅ Database reset complete — data cleared and IDs restarted from 1.")

if __name__ == "__main__":
    reset()
