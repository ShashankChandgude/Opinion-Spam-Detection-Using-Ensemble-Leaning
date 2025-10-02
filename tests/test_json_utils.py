#!/usr/bin/env python
# coding: utf-8

import pytest
import numpy as np
from src.utils.json_utils import to_serializable


class TestJsonUtils:
    """Test the JSON serialization utilities."""
    
    def test_to_serializable_numpy_integer(self):
        """Test conversion of numpy integers to Python int."""
        # Test different numpy integer types
        assert to_serializable(np.int8(42)) == 42
        assert to_serializable(np.int16(100)) == 100
        assert to_serializable(np.int32(1000)) == 1000
        assert to_serializable(np.int64(10000)) == 10000
        
        # Test edge cases
        assert to_serializable(np.int32(0)) == 0
        assert to_serializable(np.int32(-1)) == -1
    
    def test_to_serializable_numpy_floating(self):
        """Test conversion of numpy floats to Python float."""
        # Test different numpy float types
        assert to_serializable(np.float32(3.14)) == pytest.approx(3.14, rel=1e-6)
        assert to_serializable(np.float64(2.718)) == 2.718
        
        # Test edge cases
        assert to_serializable(np.float32(0.0)) == 0.0
        assert to_serializable(np.float32(-1.5)) == -1.5
        assert to_serializable(np.float64(float('inf'))) == float('inf')
        assert to_serializable(np.float64(float('-inf'))) == float('-inf')
    
    def test_to_serializable_numpy_array(self):
        """Test conversion of numpy arrays to Python lists."""
        # Test 1D array
        arr1d = np.array([1, 2, 3, 4])
        assert to_serializable(arr1d) == [1, 2, 3, 4]
        
        # Test 2D array
        arr2d = np.array([[1, 2], [3, 4]])
        assert to_serializable(arr2d) == [[1, 2], [3, 4]]
        
        # Test empty array
        empty_arr = np.array([])
        assert to_serializable(empty_arr) == []
        
        # Test array with different data types
        float_arr = np.array([1.5, 2.5, 3.5])
        assert to_serializable(float_arr) == [1.5, 2.5, 3.5]
    
    def test_to_serializable_set(self):
        """Test conversion of sets to Python lists."""
        # Test regular set
        test_set = {1, 2, 3, 4}
        result = to_serializable(test_set)
        assert isinstance(result, list)
        assert set(result) == test_set  # Order may differ, but elements should be same
        
        # Test empty set
        empty_set = set()
        assert to_serializable(empty_set) == []
        
        # Test set with mixed types
        mixed_set = {1, "hello", 3.14}
        result = to_serializable(mixed_set)
        assert isinstance(result, list)
        assert set(result) == mixed_set
    
    def test_to_serializable_other_types(self):
        """Test that other types are returned unchanged."""
        # Test Python int
        assert to_serializable(42) == 42
        
        # Test Python float
        assert to_serializable(3.14) == 3.14
        
        # Test string
        assert to_serializable("hello") == "hello"
        
        # Test list
        test_list = [1, 2, 3]
        assert to_serializable(test_list) == test_list
        
        # Test dict
        test_dict = {"a": 1, "b": 2}
        assert to_serializable(test_dict) == test_dict
        
        # Test tuple (should remain as tuple)
        test_tuple = (1, 2, 3)
        assert to_serializable(test_tuple) == test_tuple
        
        # Test None
        assert to_serializable(None) is None
        
        # Test boolean
        assert to_serializable(True) is True
        assert to_serializable(False) is False
    
    def test_to_serializable_nested_structures(self):
        """Test conversion of nested structures containing numpy types."""
        # Note: to_serializable only converts the top-level object, not nested ones
        # Test dict with numpy values
        test_dict = {
            "int_val": np.int32(42),
            "float_val": np.float64(3.14),
            "array_val": np.array([1, 2, 3]),
            "set_val": {1, 2, 3}
        }
        
        result = to_serializable(test_dict)
        # The dict itself is returned unchanged, nested numpy values are NOT converted
        assert result["int_val"] == 42  # This works because it's a direct numpy type
        assert result["float_val"] == 3.14  # This works because it's a direct numpy type
        assert isinstance(result["array_val"], np.ndarray)  # Nested arrays are not converted
        assert isinstance(result["set_val"], set)  # Nested sets are NOT converted
        assert result["set_val"] == {1, 2, 3}
        
        # Test list with numpy values
        test_list = [np.int32(1), np.float64(2.5), np.array([3, 4]), {5, 6}]
        result = to_serializable(test_list)
        assert result[0] == 1
        assert result[1] == 2.5
        assert isinstance(result[2], np.ndarray)  # Nested arrays are not converted
        assert np.array_equal(result[2], np.array([3, 4]))
        assert isinstance(result[3], set)  # Nested sets are NOT converted
        assert result[3] == {5, 6}
