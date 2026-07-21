from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text
import os
import logging

logger = logging.getLogger(__name__)

# Track primary database error for diagnosis
PRIMARY_DB_ERROR = None

# Default to SQLite for local work
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tb_system.db").strip()


# Fix for Render/Heroku style postgres URLs if needed
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

def get_engine_args(url):
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    else:
        return {
            "sslmode": "require",
            "connect_timeout": 5 # Prevent infinite socket hang
        }

# Safe engine creation with fallback to SQLite if connection fails
try:
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
        engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=get_engine_args(SQLALCHEMY_DATABASE_URL))
    else:
        try:
            # Production settings for PostgreSQL (Render/Supabase)
            engine = create_engine(
                SQLALCHEMY_DATABASE_URL,
                pool_pre_ping=True,      # Check connection health before using
                pool_recycle=300,        # Refresh connections every 5 mins to prevent Render drops
                pool_size=5,            # Keep 5 connections open
                max_overflow=10,        # Allow 10 extra temporary connections
                pool_timeout=10,        # Fail if no connection available in 10s
                connect_args=get_engine_args(SQLALCHEMY_DATABASE_URL)
            )
            # Test connection health immediately
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Primary PostgreSQL database connected successfully.")
        except Exception as conn_err:
            conn_err_str = str(conn_err).lower()
            # If the hostname couldn't be resolved, try appending Render's external regional domains
            if "could not translate host name" in conn_err_str or "name or service not known" in conn_err_str:
                regions = [
                    "oregon-postgres.render.com",
                    "frankfurt-postgres.render.com",
                    "singapore-postgres.render.com",
                    "ohio-postgres.render.com"
                ]
                resolved = False
                for region in regions:
                    try:
                        url_parts = SQLALCHEMY_DATABASE_URL.split("@")
                        if len(url_parts) == 2:
                            host_db = url_parts[1].split("/")
                            if len(host_db) == 2:
                                host = host_db[0]
                                db = host_db[1]
                                # If it's an internal Render hostname, append the regional suffix
                                if not host.endswith(".render.com"):
                                    test_url = f"{url_parts[0]}@{host}.{region}/{db}"
                                    logger.info(f"Attempting connection to external Render Postgres: {host}.{region}")
                                    test_engine = create_engine(
                                        test_url,
                                        pool_pre_ping=True,
                                        pool_recycle=300,
                                        pool_size=5,
                                        max_overflow=10,
                                        pool_timeout=10,
                                        connect_args=get_engine_args(test_url)
                                    )
                                    with test_engine.connect() as conn:
                                        conn.execute(text("SELECT 1"))
                                    # Connection succeeded! Use this engine.
                                    engine = test_engine
                                    SQLALCHEMY_DATABASE_URL = test_url
                                    logger.info(f"Successfully connected to external database in region: {region}")
                                    resolved = True
                                    break
                    except Exception as ext_err:
                        logger.warning(f"External connection to {region} failed: {ext_err}")
                
                if not resolved:
                    raise conn_err
            else:
                raise conn_err
except Exception as e:
    PRIMARY_DB_ERROR = str(e)
    logger.error(f"Primary database connection failed: {e}")
    logger.warning("FALLING BACK TO LOCAL SQLITE DATABASE FOR RESILIENCE.")
    SQLALCHEMY_DATABASE_URL = "sqlite:///./tb_system.db"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

