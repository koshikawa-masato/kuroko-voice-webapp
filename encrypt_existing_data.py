#!/usr/bin/env python3
"""
Encrypt existing data files.

Run this script ONCE after setting up ENCRYPTION_KEY in .env
to encrypt all existing user data and history files.

Can also be used to re-encrypt data after encryption format changes.
"""

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from encryption import (
    encrypt_user_data, decrypt_user_data,
    encrypt_history_data, decrypt_history_data,
    get_encryption
)

DATA_DIR = Path("/tmp/kuroko_data")
USERS_FILE = DATA_DIR / "users.json"
HISTORY_DIR = DATA_DIR / "history"


def is_old_format_encrypted(text: str) -> bool:
    """Check if text is encrypted with old double-base64 format."""
    if not text:
        return False
    try:
        # Old format: base64(base64(fernet_token))
        # When decoded once, it should start with 'gAAA'
        decoded = base64.urlsafe_b64decode(text.encode('utf-8'))
        return decoded[:4] == b'gAAA'
    except Exception:
        return False


def decrypt_old_format(text: str, enc) -> str:
    """Decrypt text from old double-base64 format."""
    try:
        # First decode the outer base64
        inner = base64.urlsafe_b64decode(text.encode('utf-8'))
        # inner is now the Fernet token, decrypt it
        decrypted = enc.fernet.decrypt(inner)
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"    Old format decrypt error: {e}")
        return text


def encrypt_users_file():
    """Encrypt existing users.json file."""
    if not USERS_FILE.exists():
        print("No users.json file found, skipping...")
        return

    print(f"Processing {USERS_FILE}...")

    with open(USERS_FILE, "r") as f:
        users = json.load(f)

    enc = get_encryption()
    encrypted_count = 0
    already_encrypted = 0
    migrated_count = 0

    for username, user_data in users.items():
        needs_save = False

        # Check each sensitive field
        for field in ['password', 'github_username']:
            if field not in user_data or not user_data[field]:
                continue

            value = user_data[field]

            # Check if it's the new correct format
            if enc.is_encrypted(value):
                already_encrypted += 1
                continue

            # Check if it's the old double-base64 format
            if is_old_format_encrypted(value):
                print(f"    Migrating {field} from old format...")
                decrypted = decrypt_old_format(value, enc)
                user_data[field] = enc.encrypt(decrypted)
                migrated_count += 1
                needs_save = True
            else:
                # Plain text, encrypt it
                user_data[field] = enc.encrypt(value)
                encrypted_count += 1
                needs_save = True

        if needs_save:
            users[username] = user_data

    # Save encrypted data
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

    print(f"  Encrypted: {encrypted_count} fields")
    print(f"  Migrated from old format: {migrated_count} fields")
    print(f"  Already encrypted (correct format): {already_encrypted} fields")


def encrypt_history_files():
    """Encrypt existing history files."""
    if not HISTORY_DIR.exists():
        print("No history directory found, skipping...")
        return

    print(f"Processing history files in {HISTORY_DIR}...")

    enc = get_encryption()
    encrypted_count = 0
    already_encrypted = 0
    error_count = 0

    # Iterate through all user directories
    for user_dir in HISTORY_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        for history_file in user_dir.glob("*.json"):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check if messages are already encrypted (string instead of list)
                if 'messages' in data and isinstance(data['messages'], str):
                    if enc.is_encrypted(data['messages']):
                        already_encrypted += 1
                        continue

                # Encrypt history data
                encrypted_data = encrypt_history_data(data)

                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(encrypted_data, f, ensure_ascii=False, indent=2)

                encrypted_count += 1

            except Exception as e:
                print(f"  Error processing {history_file}: {e}")
                error_count += 1

    print(f"  Encrypted: {encrypted_count} files")
    print(f"  Already encrypted: {already_encrypted} files")
    print(f"  Errors: {error_count} files")


def main():
    print("=" * 60)
    print("Encrypting existing data files")
    print("=" * 60)
    print()

    # Check if ENCRYPTION_KEY is set
    if not os.getenv("ENCRYPTION_KEY"):
        print("ERROR: ENCRYPTION_KEY not set in .env file!")
        print()
        print("Generate a key with:")
        print("  python encryption.py")
        print()
        print("Then add it to your .env file and run this script again.")
        return

    print("ENCRYPTION_KEY is set, proceeding...")
    print()

    encrypt_users_file()
    print()
    encrypt_history_files()
    print()

    print("=" * 60)
    print("Done! All existing data has been encrypted.")
    print("=" * 60)


if __name__ == "__main__":
    main()
