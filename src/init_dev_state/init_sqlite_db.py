from promenade.models import create_engine, Base, text, os, DATA_DIR, DATABASE_ADRESS

os.makedirs(DATA_DIR, exist_ok=True)
engine = create_engine(DATABASE_ADRESS)

Base.metadata.create_all(engine)

with engine.connect() as conn:
    conn.execute(text("DELETE FROM schedule"))
    conn.execute(text("DELETE FROM museum"))
    conn.commit()


