"""
Data Encryption - AES-256 encryption for stored data.

Provides field-level encryption for sensitive data (passwords, user info).
Based on Sisters-On-WhatsApp encryption module.
"""

import os
import base64
import hashlib
import json
import logging
from typing import Optional, Any, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class DataEncryption:
    """
    AES-256 encryption for sensitive data.

    Uses Fernet (AES-128-CBC with HMAC) which is simpler and secure enough.
    For true AES-256, we use PBKDF2 to derive a 256-bit key.
    """

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption with key from environment or parameter.

        Args:
            encryption_key: Base64-encoded encryption key, or will use ENCRYPTION_KEY env var
        """
        key_source = encryption_key or os.getenv("ENCRYPTION_KEY")

        if not key_source:
            logger.warning("ENCRYPTION_KEY not set - generating new key (NOT FOR PRODUCTION)")
            # Generate a key for development only
            key_source = Fernet.generate_key().decode()
            logger.warning(f"Generated key (save this!): {key_source}")

        # Derive a proper Fernet key from the source key
        self.fernet = self._create_fernet(key_source)

    def _create_fernet(self, key_source: str) -> Fernet:
        """Create Fernet instance from key source."""
        # If it's already a valid Fernet key (44 chars base64), use directly
        if len(key_source) == 44:
            try:
                return Fernet(key_source.encode())
            except Exception:
                pass

        # Otherwise, derive a key using PBKDF2
        salt = b"kuroko_interview_v1"  # Static salt (key is already secret)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_source.encode()))
        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string.

        Args:
            plaintext: The text to encrypt

        Returns:
            Fernet encrypted string (already base64 encoded)
        """
        if not plaintext:
            return plaintext

        try:
            encrypted = self.fernet.encrypt(plaintext.encode('utf-8'))
            # Fernet output is already base64, return as string
            return encrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext string.

        Args:
            ciphertext: Fernet encrypted string

        Returns:
            Decrypted plaintext
        """
        if not ciphertext:
            return ciphertext

        try:
            decrypted = self.fernet.decrypt(ciphertext.encode('utf-8'))
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            # Return original if decryption fails (might be unencrypted legacy data)
            return ciphertext

    def is_encrypted(self, text: str) -> bool:
        """Check if text appears to be Fernet encrypted."""
        if not text:
            return False

        # Fernet tokens start with 'gAAA' (base64 of version byte 0x80)
        if not text.startswith('gAAA'):
            return False

        try:
            # Try to decode as base64 to verify format
            decoded = base64.urlsafe_b64decode(text.encode('utf-8'))
            # Fernet tokens start with version byte 0x80
            return len(decoded) > 0 and decoded[0:1] == b'\x80'
        except Exception:
            return False

    def encrypt_if_needed(self, text: str) -> str:
        """Encrypt text only if not already encrypted."""
        if self.is_encrypted(text):
            return text
        return self.encrypt(text)

    def decrypt_if_needed(self, text: str) -> str:
        """Decrypt text only if it's encrypted."""
        if not self.is_encrypted(text):
            return text
        return self.decrypt(text)

    def encrypt_json(self, data: Union[dict, list]) -> str:
        """Encrypt a JSON-serializable object."""
        if data is None:
            return None
        json_str = json.dumps(data, ensure_ascii=False)
        return self.encrypt(json_str)

    def decrypt_json(self, ciphertext: str) -> Union[dict, list, None]:
        """Decrypt to a JSON object."""
        if not ciphertext:
            return None

        decrypted = self.decrypt_if_needed(ciphertext)

        # If decryption returned original (legacy unencrypted data)
        if decrypted == ciphertext:
            # Try parsing as JSON directly
            try:
                return json.loads(ciphertext)
            except json.JSONDecodeError:
                return ciphertext

        try:
            return json.loads(decrypted)
        except json.JSONDecodeError:
            return decrypted


# Singleton instance
_encryption = None


def get_encryption() -> DataEncryption:
    """Get singleton encryption instance."""
    global _encryption
    if _encryption is None:
        _encryption = DataEncryption()
    return _encryption


def encrypt_user_data(data: dict) -> dict:
    """
    Encrypt sensitive fields in user data for storage.

    Encrypts: password (already hashed but encrypt for extra security),
              github_username, and any other sensitive fields.
    """
    enc = get_encryption()
    encrypted = data.copy()

    # Fields to encrypt
    sensitive_fields = ['password', 'github_username']

    for field in sensitive_fields:
        if field in encrypted and encrypted[field]:
            encrypted[field] = enc.encrypt_if_needed(str(encrypted[field]))

    return encrypted


def decrypt_user_data(data: dict) -> dict:
    """
    Decrypt sensitive fields in user data after retrieval.
    """
    enc = get_encryption()
    decrypted = data.copy()

    # Fields to decrypt
    sensitive_fields = ['password', 'github_username']

    for field in sensitive_fields:
        if field in decrypted and decrypted[field]:
            decrypted[field] = enc.decrypt_if_needed(str(decrypted[field]))

    return decrypted


def encrypt_history_data(data: dict) -> dict:
    """
    Encrypt sensitive fields in history data for storage.
    """
    enc = get_encryption()
    encrypted = data.copy()

    # Encrypt messages content
    if 'messages' in encrypted and encrypted['messages']:
        encrypted['messages'] = enc.encrypt_json(encrypted['messages'])

    # Encrypt score
    if 'score' in encrypted and encrypted['score']:
        encrypted['score'] = enc.encrypt_if_needed(str(encrypted['score']))

    return encrypted


def decrypt_history_data(data: dict) -> dict:
    """
    Decrypt sensitive fields in history data after retrieval.
    """
    enc = get_encryption()
    decrypted = data.copy()

    # Decrypt messages content
    if 'messages' in decrypted and decrypted['messages']:
        messages = enc.decrypt_json(decrypted['messages'])
        if messages:
            decrypted['messages'] = messages

    # Decrypt score
    if 'score' in decrypted and decrypted['score']:
        decrypted['score'] = enc.decrypt_if_needed(str(decrypted['score']))

    return decrypted


def generate_encryption_key() -> str:
    """Generate a new encryption key for .env file."""
    key = Fernet.generate_key()
    return key.decode()


if __name__ == "__main__":
    # Generate keys when run directly
    print("=" * 60)
    print("Encryption Key for .env")
    print("=" * 60)
    print()
    print("# Add this to your .env file:")
    print(f"ENCRYPTION_KEY={generate_encryption_key()}")
    print()
    print("=" * 60)

    # Demo
    print("\nDemo: Encryption/Decryption")
    print("-" * 40)
    enc = DataEncryption()
    test_data = "secret_password_123"
    encrypted = enc.encrypt(test_data)
    print(f"Original:  {test_data}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {enc.decrypt(encrypted)}")
