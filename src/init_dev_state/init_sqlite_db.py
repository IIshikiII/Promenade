from promenade.models import create_engine, Base, text, os

os.makedirs("data", exist_ok=True)
engine = create_engine("sqlite:///data/museum.db")

Base.metadata.create_all(engine)

with engine.connect() as conn:
    conn.execute(text("DELETE FROM schedule"))
    conn.execute(text("DELETE FROM museum"))
    conn.commit()


