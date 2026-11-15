"""Tests for result_cache module.

Tests cover:
- Basic put/get cache operations
- Cache hit/miss behavior with same and different signatures
- LRU eviction at max_entries_per_op boundary
- Staleness detection with is_stale()
- Cache statistics tracking
- Operation-specific cache clearing
- NaN and Inf handling in matrices
"""

import time
import pytest
import numpy as np

from src.matrix_calculator.result_cache import (
    ResultCache,
    compute_input_signature,
    _serialize_matrix,
)


class TestSerializeMatrix:
    """Tests for _serialize_matrix helper function."""
    
    def test_serialize_simple_matrix(self):
        """Test serialization of a simple 2x2 matrix."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        serialized = _serialize_matrix(matrix)
        
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
    
    def test_serialize_converts_to_float(self):
        """Test that serialization converts integer matrices to float."""
        matrix_int = np.array([[1, 2], [3, 4]], dtype=int)
        matrix_float = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        
        serialized_int = _serialize_matrix(matrix_int)
        serialized_float = _serialize_matrix(matrix_float)
        
        assert serialized_int == serialized_float
    
    def test_serialize_nan_inf(self):
        """Test serialization of matrices with NaN and Inf."""
        matrix_nan = np.array([[np.nan, 1.0], [2.0, 3.0]])
        matrix_inf = np.array([[np.inf, 1.0], [2.0, -np.inf]])
        
        serialized_nan = _serialize_matrix(matrix_nan)
        serialized_inf = _serialize_matrix(matrix_inf)
        
        assert isinstance(serialized_nan, bytes)
        assert isinstance(serialized_inf, bytes)
    
    def test_serialize_deterministic(self):
        """Test that serialization is deterministic."""
        matrix = np.array([[1.5, 2.7], [3.2, 4.1]])
        
        serialized1 = _serialize_matrix(matrix)
        serialized2 = _serialize_matrix(matrix)
        
        assert serialized1 == serialized2


class TestComputeInputSignature:
    """Tests for compute_input_signature helper function."""
    
    def test_signature_with_single_matrix(self):
        """Test signature computation with a single matrix."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        sig = compute_input_signature([matrix], "add")
        
        assert isinstance(sig, tuple)
        assert len(sig) == 3
        assert sig[0] == "add"  # operation_code
        assert isinstance(sig[1], tuple)  # matrix_ids
        assert isinstance(sig[2], str)  # hash_string
    
    def test_signature_with_multiple_matrices(self):
        """Test signature computation with multiple matrices."""
        matrix_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix_b = np.array([[5.0, 6.0], [7.0, 8.0]])
        sig = compute_input_signature([matrix_a, matrix_b], "multiply")
        
        assert len(sig) == 3
        assert sig[0] == "multiply"
    
    def test_signature_deterministic(self):
        """Test that signatures are deterministic."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        sig1 = compute_input_signature([matrix], "add")
        sig2 = compute_input_signature([matrix], "add")
        
        assert sig1 == sig2
    
    def test_signature_different_for_different_matrices(self):
        """Test that different matrices produce different signatures."""
        matrix_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix_b = np.array([[1.0, 2.0], [3.0, 5.0]])  # Different element
        
        sig_a = compute_input_signature([matrix_a], "add")
        sig_b = compute_input_signature([matrix_b], "add")
        
        assert sig_a != sig_b
    
    def test_signature_different_for_different_operations(self):
        """Test that same matrix with different operations produce different signatures."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        sig_add = compute_input_signature([matrix], "add")
        sig_multiply = compute_input_signature([matrix], "multiply")
        
        # Operation code is part of the signature, so they should differ
        assert sig_add != sig_multiply
    
    def test_signature_with_empty_list(self):
        """Test signature computation with empty matrix list."""
        sig = compute_input_signature([], "determinant")
        
        assert isinstance(sig, tuple)
        assert sig[0] == "determinant"
    
    def test_signature_different_shapes_same_data(self):
        """Test that matrices with different shapes produce different signatures."""
        matrix_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix_1x4 = np.array([[1.0, 2.0, 3.0, 4.0]])
        
        sig_2x2 = compute_input_signature([matrix_2x2], "test")
        sig_1x4 = compute_input_signature([matrix_1x4], "test")
        
        # Different shapes should produce different matrix_ids
        assert sig_2x2[1] != sig_1x4[1] or sig_2x2[2] != sig_1x4[2]
    
    def test_signature_nan_handling(self):
        """Test signature computation with NaN values."""
        matrix_nan = np.array([[np.nan, 1.0], [2.0, 3.0]])
        sig = compute_input_signature([matrix_nan], "test")
        
        assert isinstance(sig, tuple)
        assert isinstance(sig[2], str)


class TestResultCacheBasics:
    """Tests for basic ResultCache operations."""
    
    def test_cache_initialization(self):
        """Test cache initialization with default and custom max_entries."""
        cache1 = ResultCache()
        assert cache1.max_entries_per_op == 10
        
        cache2 = ResultCache(max_entries_per_op=5)
        assert cache2.max_entries_per_op == 5
    
    def test_put_and_get_basic(self):
        """Test basic put and get operations."""
        cache = ResultCache()
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        sig = compute_input_signature([matrix], "add")
        
        # Initially, cache should be empty
        assert cache.get("add", sig) is None
        
        # Put entry in cache
        result = np.array([[2.0, 3.0], [4.0, 5.0]])
        cache.put("add", sig, result)
        
        # Now get should return the entry
        entry = cache.get("add", sig)
        assert entry is not None
        assert np.array_equal(entry["data"], result)
    
    def test_get_returns_copy(self):
        """Test that cached data is a copy, not a reference."""
        cache = ResultCache()
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        sig = compute_input_signature([matrix], "add")
        
        result = np.array([[2.0, 3.0], [4.0, 5.0]])
        cache.put("add", sig, result)
        
        entry = cache.get("add", sig)
        # Modify the returned data
        entry["data"][0, 0] = 999.0
        
        # Get again and verify original is unchanged
        entry2 = cache.get("add", sig)
        assert entry2["data"][0, 0] != 999.0
    
    def test_cache_entry_structure(self):
        """Test that cache entry contains all required fields."""
        cache = ResultCache()
        matrix = np.array([[1.0, 2.0]])
        sig = compute_input_signature([matrix], "add")
        
        result = np.array([[2.0, 3.0]])
        cache.put("add", sig, result)
        
        entry = cache.get("add", sig)
        
        assert "data" in entry
        assert "timestamp" in entry
        assert "hit_count" in entry
        assert "input_signature" in entry
        assert "operation_code" in entry
        assert entry["operation_code"] == "add"


class TestCacheHitMiss:
    """Tests for cache hit/miss behavior and statistics."""
    
    def test_cache_hit_increments_counter(self):
        """Test that cache hits increment the hit counter."""
        cache = ResultCache()
        matrix = np.array([[1.0, 2.0]])
        sig = compute_input_signature([matrix], "add")
        
        result = np.array([[2.0, 3.0]])
        cache.put("add", sig, result)
        
        # First get is a hit
        entry = cache.get("add", sig)
        assert entry["hit_count"] == 1
        
        # Second get increments hit count
        entry = cache.get("add", sig)
        assert entry["hit_count"] == 2
    
    def test_cache_miss_increments_counter(self):
        """Test that cache misses are tracked."""
        cache = ResultCache()
        stats = cache.get_stats("add")
        assert stats["miss_count"] == 0
        
        # Miss on non-existent entry
        sig = compute_input_signature([np.array([[1.0]])], "add")
        cache.get("add", sig)
        
        stats = cache.get_stats("add")
        assert stats["miss_count"] == 1
    
    def test_hit_and_miss_counts(self):
        """Test that hit and miss counts are tracked correctly."""
        cache = ResultCache()
        matrix = np.array([[1.0, 2.0]])
        sig1 = compute_input_signature([matrix], "add")
        sig2 = compute_input_signature([matrix * 2], "add")
        
        result = np.array([[2.0, 3.0]])
        cache.put("add", sig1, result)
        
        # 1 hit
        cache.get("add", sig1)
        # 1 miss
        cache.get("add", sig2)
        # 1 more hit
        cache.get("add", sig1)
        
        stats = cache.get_stats("add")
        assert stats["hit_count"] == 2
        assert stats["miss_count"] == 1


class TestLRUEviction:
    """Tests for LRU eviction at max_entries_per_op boundary."""
    
    def test_lru_eviction_at_max_entries(self):
        """Test that oldest entry is evicted when max_entries_per_op is exceeded."""
        cache = ResultCache(max_entries_per_op=3)
        
        # Add 3 entries
        for i in range(3):
            matrix = np.array([[float(i)]])
            sig = compute_input_signature([matrix], "add")
            result = np.array([[float(i + 10)]])
            cache.put("add", sig, result)
        
        stats = cache.get_stats("add")
        assert stats["total_entries"] == 3
        
        # Add 4th entry; should evict oldest
        matrix4 = np.array([[3.0]])
        sig4 = compute_input_signature([matrix4], "add")
        result4 = np.array([[13.0]])
        cache.put("add", sig4, result4)
        
        stats = cache.get_stats("add")
        assert stats["total_entries"] == 3  # Still 3, oldest was evicted
    
    def test_first_entry_evicted_on_overflow(self):
        """Test that first (oldest) entry is evicted, not random."""
        cache = ResultCache(max_entries_per_op=2)
        
        # Add entry 1
        sig1 = compute_input_signature([np.array([[1.0]])], "test")
        cache.put("test", sig1, np.array([[10.0]]))
        
        # Add entry 2
        sig2 = compute_input_signature([np.array([[2.0]])], "test")
        cache.put("test", sig2, np.array([[20.0]]))
        
        # Verify both are cached
        assert cache.get("test", sig1) is not None
        assert cache.get("test", sig2) is not None
        
        # Add entry 3; should evict sig1 (oldest)
        sig3 = compute_input_signature([np.array([[3.0]])], "test")
        cache.put("test", sig3, np.array([[30.0]]))
        
        # sig1 should be gone, sig2 and sig3 should remain
        assert cache.get("test", sig1) is None  # Evicted
        assert cache.get("test", sig2) is not None
        assert cache.get("test", sig3) is not None
    
    def test_lru_per_operation(self):
        """Test that LRU eviction is per-operation."""
        cache = ResultCache(max_entries_per_op=2)
        
        # Fill "add" cache
        sig_add_1 = compute_input_signature([np.array([[1.0]])], "add")
        cache.put("add", sig_add_1, np.array([[10.0]]))
        sig_add_2 = compute_input_signature([np.array([[2.0]])], "add")
        cache.put("add", sig_add_2, np.array([[20.0]]))
        
        # Fill "multiply" cache
        sig_mul_1 = compute_input_signature([np.array([[1.0]])], "multiply")
        cache.put("multiply", sig_mul_1, np.array([[10.0]]))
        sig_mul_2 = compute_input_signature([np.array([[2.0]])], "multiply")
        cache.put("multiply", sig_mul_2, np.array([[20.0]]))
        
        # Add 3rd entry to "add"; should evict oldest "add" entry only
        sig_add_3 = compute_input_signature([np.array([[3.0]])], "add")
        cache.put("add", sig_add_3, np.array([[30.0]]))
        
        # Verify add cache
        assert cache.get("add", sig_add_1) is None  # Evicted
        assert cache.get("add", sig_add_2) is not None
        assert cache.get("add", sig_add_3) is not None
        
        # Verify multiply cache untouched
        assert cache.get("multiply", sig_mul_1) is not None
        assert cache.get("multiply", sig_mul_2) is not None


class TestStalenessDetection:
    """Tests for is_stale() staleness detection."""
    
    def test_is_stale_no_cache(self):
        """Test is_stale returns True when no cache exists."""
        cache = ResultCache()
        sig = compute_input_signature([np.array([[1.0]])], "add")
        
        assert cache.is_stale("add", sig) is True
    
    def test_is_stale_cache_exists_same_sig(self):
        """Test is_stale returns False when cached entry has same signature."""
        cache = ResultCache()
        sig = compute_input_signature([np.array([[1.0]])], "add")
        
        cache.put("add", sig, np.array([[2.0]]))
        
        # Same signature should not be stale
        assert cache.is_stale("add", sig) is False
    
    def test_is_stale_cache_exists_different_sig(self):
        """Test is_stale returns True when current signature differs."""
        cache = ResultCache()
        sig1 = compute_input_signature([np.array([[1.0]])], "add")
        sig2 = compute_input_signature([np.array([[2.0]])], "add")
        
        cache.put("add", sig1, np.array([[10.0]]))
        
        # Different signature should be stale
        assert cache.is_stale("add", sig2) is True
    
    def test_is_stale_multiple_cached_entries(self):
        """Test is_stale with multiple cached entries for same operation."""
        cache = ResultCache()
        sig1 = compute_input_signature([np.array([[1.0]])], "add")
        sig2 = compute_input_signature([np.array([[2.0]])], "add")
        sig3 = compute_input_signature([np.array([[3.0]])], "add")
        
        cache.put("add", sig1, np.array([[10.0]]))
        cache.put("add", sig2, np.array([[20.0]]))
        
        # sig1 and sig2 should not be stale
        assert cache.is_stale("add", sig1) is False
        assert cache.is_stale("add", sig2) is False
        
        # sig3 should be stale
        assert cache.is_stale("add", sig3) is True


class TestCacheStatistics:
    """Tests for cache statistics tracking."""
    
    def test_get_stats_empty_cache(self):
        """Test get_stats on empty cache."""
        cache = ResultCache()
        stats = cache.get_stats("add")
        
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["total_entries"] == 0
    
    def test_get_stats_with_entries(self):
        """Test get_stats reflects current cache state."""
        cache = ResultCache()
        sig = compute_input_signature([np.array([[1.0]])], "add")
        
        cache.put("add", sig, np.array([[2.0]]))
        cache.get("add", sig)  # hit
        cache.get("add", compute_input_signature([np.array([[2.0]])], "add"))  # miss
        
        stats = cache.get_stats("add")
        assert stats["total_entries"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
    
    def test_stats_per_operation(self):
        """Test that statistics are tracked per operation."""
        cache = ResultCache()
        
        sig_add = compute_input_signature([np.array([[1.0]])], "add")
        sig_mul = compute_input_signature([np.array([[1.0]])], "multiply")
        
        cache.put("add", sig_add, np.array([[2.0]]))
        cache.put("multiply", sig_mul, np.array([[2.0]]))
        
        cache.get("add", sig_add)  # hit
        cache.get("add", sig_add)  # hit
        cache.get("multiply", sig_mul)  # hit
        
        stats_add = cache.get_stats("add")
        stats_mul = cache.get_stats("multiply")
        
        assert stats_add["hit_count"] == 2
        assert stats_mul["hit_count"] == 1


class TestCacheClearing:
    """Tests for clear() and clear_operation() methods."""
    
    def test_clear_all_entries(self):
        """Test clear() removes all cache entries and resets stats."""
        cache = ResultCache()
        
        sig_add = compute_input_signature([np.array([[1.0]])], "add")
        sig_mul = compute_input_signature([np.array([[1.0]])], "multiply")
        
        cache.put("add", sig_add, np.array([[2.0]]))
        cache.put("multiply", sig_mul, np.array([[2.0]]))
        
        # Verify entries exist
        assert cache.get("add", sig_add) is not None
        assert cache.get("multiply", sig_mul) is not None
        
        # Clear all
        cache.clear()
        
        # Verify all gone
        assert cache.get("add", sig_add) is None
        assert cache.get("multiply", sig_mul) is None
        assert cache.get_stats("add")["total_entries"] == 0
        assert cache.get_stats("multiply")["total_entries"] == 0
    
    def test_clear_operation_specific(self):
        """Test clear_operation() removes entries for one operation only."""
        cache = ResultCache()
        
        sig_add = compute_input_signature([np.array([[1.0]])], "add")
        sig_mul = compute_input_signature([np.array([[1.0]])], "multiply")
        
        cache.put("add", sig_add, np.array([[2.0]]))
        cache.put("multiply", sig_mul, np.array([[2.0]]))
        
        # Clear only "add"
        cache.clear_operation("add")
        
        # "add" should be gone, "multiply" should remain
        assert cache.get("add", sig_add) is None
        assert cache.get("multiply", sig_mul) is not None
    
    def test_clear_operation_resets_stats(self):
        """Test that clear_operation() resets statistics."""
        cache = ResultCache()
        sig = compute_input_signature([np.array([[1.0]])], "add")
        
        cache.put("add", sig, np.array([[2.0]]))
        cache.get("add", sig)  # hit
        cache.get("add", compute_input_signature([np.array([[2.0]])], "add"))  # miss
        
        stats_before = cache.get_stats("add")
        assert stats_before["hit_count"] == 1
        assert stats_before["miss_count"] == 1
        
        cache.clear_operation("add")
        
        stats_after = cache.get_stats("add")
        assert stats_after["hit_count"] == 0
        assert stats_after["miss_count"] == 0


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_cache_with_nan_matrices(self):
        """Test caching matrices with NaN values."""
        cache = ResultCache()
        
        matrix_with_nan = np.array([[np.nan, 1.0], [2.0, 3.0]])
        sig = compute_input_signature([matrix_with_nan], "test")
        
        result = np.array([[1.0, 2.0], [3.0, 4.0]])
        cache.put("test", sig, result)
        
        entry = cache.get("test", sig)
        assert entry is not None
        assert np.array_equal(entry["data"], result, equal_nan=True)
    
    def test_cache_with_inf_matrices(self):
        """Test caching matrices with Inf values."""
        cache = ResultCache()
        
        matrix_with_inf = np.array([[np.inf, 1.0], [2.0, -np.inf]])
        sig = compute_input_signature([matrix_with_inf], "test")
        
        result = np.array([[1.0, 2.0], [3.0, 4.0]])
        cache.put("test", sig, result)
        
        entry = cache.get("test", sig)
        assert entry is not None
        assert np.array_equal(entry["data"], result)
    
    def test_cache_timestamp_is_set(self):
        """Test that cache entry timestamp is set correctly."""
        cache = ResultCache()
        sig = compute_input_signature([np.array([[1.0]])], "test")
        
        before = time.time()
        cache.put("test", sig, np.array([[2.0]]))
        after = time.time()
        
        entry = cache.get("test", sig)
        timestamp = entry["timestamp"]
        
        assert before <= timestamp <= after
    
    def test_cache_with_large_matrix(self):
        """Test caching a large matrix."""
        cache = ResultCache()
        
        large_matrix = np.random.rand(100, 100)
        sig = compute_input_signature([large_matrix], "test")
        
        result = np.random.rand(100, 100)
        cache.put("test", sig, result)
        
        entry = cache.get("test", sig)
        assert entry is not None
        assert np.array_equal(entry["data"], result)
    
    def test_cache_with_rectangular_matrices(self):
        """Test caching rectangular (non-square) matrices."""
        cache = ResultCache()
        
        matrix_rect = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sig = compute_input_signature([matrix_rect], "test")
        
        result = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        cache.put("test", sig, result)
        
        entry = cache.get("test", sig)
        assert entry is not None
        assert entry["data"].shape == (2, 3)
    
    def test_cache_with_single_element_matrix(self):
        """Test caching a 1x1 matrix (scalar result)."""
        cache = ResultCache()
        
        matrix = np.array([[1.0]])
        sig = compute_input_signature([matrix], "test")
        
        result = np.array([[42.0]])
        cache.put("test", sig, result)
        
        entry = cache.get("test", sig)
        assert entry is not None
        assert entry["data"][0, 0] == 42.0
    
    def test_multiple_signatures_same_operation(self):
        """Test handling multiple different signatures for same operation."""
        cache = ResultCache()
        
        sigs = []
        for i in range(5):
            matrix = np.array([[float(i)]])
            sig = compute_input_signature([matrix], "add")
            sigs.append(sig)
            cache.put("add", sig, np.array([[float(i + 10)]]))
        
        # All should be retrievable
        for i, sig in enumerate(sigs):
            entry = cache.get("add", sig)
            assert entry is not None
            assert entry["data"][0, 0] == float(i + 10)
    
    def test_cache_identical_operation_different_matrices(self):
        """Test that identical operations on different matrices have different cache entries."""
        cache = ResultCache()
        
        matrix1 = np.array([[1.0, 2.0]])
        matrix2 = np.array([[3.0, 4.0]])
        
        sig1 = compute_input_signature([matrix1], "determinant")
        sig2 = compute_input_signature([matrix2], "determinant")
        
        result1 = np.array([[10.0]])
        result2 = np.array([[20.0]])
        
        cache.put("determinant", sig1, result1)
        cache.put("determinant", sig2, result2)
        
        entry1 = cache.get("determinant", sig1)
        entry2 = cache.get("determinant", sig2)
        
        assert entry1["data"][0, 0] == 10.0
        assert entry2["data"][0, 0] == 20.0
