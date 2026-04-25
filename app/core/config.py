# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

DEFAULT_START_DATE = "2022-04-04"
MODEL_DIR = "app/models"