"""Result caching module for session-local computation result storage.

This module provides a thread-safe cache for storing and retrieving computation results
with staleness detection based on input signatures. The cache helps avoid redundant
computations when inputs haven't changed.

Key Features:
- Session-local caching per operation type
- LRU eviction when max_entries_per_op is exceeded
- Input signature computation via SHA-256 hashing
- Staleness detection by comparing signatures
- Cache statistics tracking (hit_count, miss_count, total_entries)
- Framework-agnostic design (no Qt, Streamlit, or other framework imports)
"""

import collections
import hashlib
import time
from typing import Dict, Optional, Tuple, Any

import numpy as np


def _serialize_matrix(matrix: np.ndarray) -> bytes:
    """Serialize a matrix to bytes using numpy tobytes().
    
    Args:
        matrix: numpy array to serialize
        
    Returns:
        bytes representation of the matrix using C-order (row-major)
    """
    return np.array(matrix, dtype=float).tobytes()


def compute_input_signature(
    matrices_list: list,
    operation_code: str,
) -> Tuple[str, Tuple[int, ...], str]:
    """Compute a deterministic input signature for cache key generation.
    
    The signature is computed by:
    1. Serializing each matrix to bytes
    2. Concatenating serialized data with operation_code as delimiter
    3. Hashing with SHA-256
    
    Args:
        matrices_list: List of numpy arrays to hash
        operation_code: The operation being performed (e.g., "add", "multiply")
        
    Returns:
        Tuple of (operation_code, matrix_ids, hash_string) where:
        - operation_code: str, the input operation code
        - matrix_ids: tuple of integers representing matrix shapes
        - hash_string: SHA-256 hexdigest of serialized inputs
    """
    if not matrices_list:
        matrices_list = []
    
    # Serialize each matrix
    serialized_parts = []
    matrix_ids = []
    
    for matrix in matrices_list:
        arr = np.array(matrix, dtype=float)
        serialized_parts.append(_serialize_matrix(arr))
        # Use shape as part of matrix identifier
        matrix_ids.append(hash(arr.shape))
    
    # Concatenate with operation code as delimiter
    delimiter = operation_code.encode()
    concatenated = delimiter
    for part in serialized_parts:
        concatenated += part + delimiter
    
    # Hash the concatenated bytes
    hash_string = hashlib.sha256(concatenated).hexdigest()
    
    return (operation_code, tuple(matrix_ids), hash_string)


class ResultCache:
    """Session-local cache for computation results with LRU eviction.
    
    The cache stores computation results keyed by (operation_code, input_signature).
    When max_entries_per_op is exceeded for an operation, the oldest entry is evicted.
    
    Attributes:
        max_entries_per_op: Maximum number of entries to cache per operation type
        _cache: Dict mapping operation_code to OrderedDict of cached entries
        _stats: Dict mapping operation_code to statistics (hit_count, miss_count, total_entries)
    """
    
    def __init__(self, max_entries_per_op: int = 10):
        """Initialize the result cache.
        
        Args:
            max_entries_per_op: Maximum number of entries to cache per operation type
        """
        self.max_entries_per_op = max_entries_per_op
        self._cache: Dict[str, collections.OrderedDict] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
    
    def put(
        self,
        operation_code: str,
        input_signature: Tuple,
        data: np.ndarray,
    ) -> None:
        """Add or update a cache entry.
        
        If the cache for this operation_code exceeds max_entries_per_op,
        the oldest entry (first in OrderedDict) is evicted using LRU.
        
        Args:
            operation_code: The operation code (e.g., "add", "multiply")
            input_signature: Tuple of (op_code, matrix_ids, hash_string)
            data: The numpy array result to cache
        """
        # Initialize operation cache if needed
        if operation_code not in self._cache:
            self._cache[operation_code] = collections.OrderedDict()
            self._stats[operation_code] = {
                "hit_count": 0,
                "miss_count": 0,
                "total_entries": 0,
            }
        
        op_cache = self._cache[operation_code]
        
        # Evict LRU entry if at capacity
        if len(op_cache) >= self.max_entries_per_op:
            # Remove the first (oldest) entry
            oldest_key = next(iter(op_cache))
            del op_cache[oldest_key]
        
        # Store new entry with metadata
        cache_entry = {
            "data": np.array(data, copy=True),  # Make a copy to avoid reference issues
            "timestamp": time.time(),
            "hit_count": 0,
            "input_signature": input_signature,
            "operation_code": operation_code,
        }
        
        op_cache[input_signature] = cache_entry
        self._stats[operation_code]["total_entries"] = len(op_cache)
    
    def get(
        self,
        operation_code: str,
        input_signature: Tuple,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result if it exists and is valid.
        
        Args:
            operation_code: The operation code (e.g., "add", "multiply")
            input_signature: Tuple of (op_code, matrix_ids, hash_string)
            
        Returns:
            Dict with cache entry (data, timestamp, hit_count, etc.) if found,
            None if not in cache
        """
        if operation_code not in self._cache:
            if operation_code not in self._stats:
                self._stats[operation_code] = {
                    "hit_count": 0,
                    "miss_count": 0,
                    "total_entries": 0,
                }
            self._stats[operation_code]["miss_count"] += 1
            return None
        
        op_cache = self._cache[operation_code]
        
        if input_signature not in op_cache:
            self._stats[operation_code]["miss_count"] += 1
            return None
        
        # Cache hit: increment counters and return a copy of the entry
        entry = op_cache[input_signature]
        entry["hit_count"] += 1
        self._stats[operation_code]["hit_count"] += 1
        
        # Return a copy of the entry with a copy of the data array
        return {
            "data": np.array(entry["data"], copy=True),
            "timestamp": entry["timestamp"],
            "hit_count": entry["hit_count"],
            "input_signature": entry["input_signature"],
            "operation_code": entry["operation_code"],
        }
    
    def is_stale(
        self,
        operation_code: str,
        current_signature: Tuple,
    ) -> bool:
        """Check if a cached result is stale due to input change.
        
        A result is stale if the current input signature differs from
        the cached input signature for the operation.
        
        Args:
            operation_code: The operation code (e.g., "add", "multiply")
            current_signature: Current input signature tuple
            
        Returns:
            True if no cache entry exists or if cached signature differs from current
            False if cached result is fresh
        """
        if operation_code not in self._cache:
            return True  # No cache, so "stale" (not cached)
        
        op_cache = self._cache[operation_code]
        
        # If we have any cached entries for this operation,
        # check if current signature matches any of them
        for cached_signature in op_cache:
            if cached_signature == current_signature:
                return False  # Found matching fresh cache
        
        return True  # No matching cache, so stale
    
    def get_stats(self, operation_code: str) -> Dict[str, Any]:
        """Get cache statistics for an operation.
        
        Args:
            operation_code: The operation code to get stats for
            
        Returns:
            Dict with keys: hit_count, miss_count, total_entries
        """
        if operation_code not in self._stats:
            return {
                "hit_count": 0,
                "miss_count": 0,
                "total_entries": 0,
            }
        
        stats = self._stats[operation_code].copy()
        stats["total_entries"] = len(
            self._cache.get(operation_code, {})
        )
        return stats
    
    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        self._cache.clear()
        self._stats.clear()
    
    def clear_operation(self, operation_code: str) -> None:
        """Clear cache entries for a specific operation.
        
        Args:
            operation_code: The operation code whose cache to clear
        """
        if operation_code in self._cache:
            self._cache[operation_code].clear()
        if operation_code in self._stats:
            self._stats[operation_code] = {
                "hit_count": 0,
                "miss_count": 0,
                "total_entries": 0,
            }
