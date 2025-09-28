from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, echo=True)  # echo=True muestra queries en consola

# Inicializa todas las tablas
def init_db():
    SQLModel.metadata.create_all(engine)

# Crea una sesión
def get_session():
    with Session(engine) as session:
        yield session
