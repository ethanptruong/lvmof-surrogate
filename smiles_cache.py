"""
smiles_cache.py — Persistent SMILES-to-feature cache.

Stores {(namespace, smiles): np.ndarray} on disk so that expensive per-SMILES
computations (ChemBERTa, DRFP, SOAP, Mordred, etc.) are only run once per
unique SMILES.  Results are reused across featurization runs as long as the
SMILES has been seen before.

Usage
-----
    from smiles_cache import get_smiles_cache

    cache = get_smiles_cache()
    vec = cache.get("chemberta_v1", smiles)   # None if not cached
    cache.set("chemberta_v1", smiles, vec)
    cache.flush()                             # persist to disk

Each namespace ends with a version tag (e.g. "_v1") so that if the
featurization logic for that block changes, bumping the version automatically
invalidates old entries without touching other namespaces.
"""

import os
import joblib

CACHE_PATH = os.path.join("checkpoints", "smiles_cache.pkl")


class SMILESCache:
    def __init__(self, path=CACHE_PATH):
        self.path = path
        self._dirty = False
        if os.path.exists(path):
            try:
                self._store = joblib.load(path)
                print(f"[smiles_cache] Loaded {len(self._store):,} entries from {path}")
            except Exception as e:
                print(f"[smiles_cache] Could not load cache ({e}) — starting fresh.")
                self._store = {}
        else:
            self._store = {}

    # ------------------------------------------------------------------
    def get(self, namespace: str, key: str):
        """Return cached value or None."""
        return self._store.get((namespace, key))

    def set(self, namespace: str, key: str, value) -> None:
        """Store value and mark cache as dirty."""
        self._store[(namespace, key)] = value
        self._dirty = True

    def flush(self) -> None:
        """Write to disk only if there are new entries."""
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        joblib.dump(self._store, self.path)
        print(f"[smiles_cache] Saved {len(self._store):,} entries → {self.path}")
        self._dirty = False

    def __len__(self) -> int:
        return len(self._store)


# Module-level singleton — loaded once per process.
_instance: SMILESCache | None = None


def get_smiles_cache() -> SMILESCache:
    global _instance
    if _instance is None:
        _instance = SMILESCache()
    return _instance
