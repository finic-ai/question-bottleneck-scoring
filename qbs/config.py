import getpass
import json
import os
from pathlib import Path

from .models import Grayson

PATH = Path(os.environ.get("QBS_CONFIG", Path.home() / ".config" / "qbs" / "config.json"))
SIGNUP_URL = "https://portal.finic.ai/keys"


def load():
    if os.environ.get("FINIC_API_KEY"):
        return {"model": "grayson", "api_key": os.environ["FINIC_API_KEY"]}
    if PATH.exists():
        return json.loads(PATH.read_text())
    return setup()


def save(config):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(config, indent=2))
    PATH.chmod(0o600)


def setup():
    print("Which decision model should answer the questions?")
    print("  1. Finic Grayson (default)")
    print("  2. Other")
    if input("Choose [1]: ").strip() == "2":
        config = {"model": "other"}
        save(config)
        print("Implement Other.answer in qbs/models.py to call your model. Keep any API key in an environment variable.")
        return config
    print(f"Get a Grayson API key at {SIGNUP_URL}")
    while True:
        key = getpass.getpass("API key: ").strip()
        try:
            Grayson(key).answer({"note": "connection check"}, ["Is this a connection check?"])
            break
        except Exception as e:
            print(f"That key didn't work: {e}")
    config = {"model": "grayson", "api_key": key}
    save(config)
    print(f"Saved to {PATH}")
    return config
