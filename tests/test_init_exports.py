"""Tests for matrix_calculator package exports and import availability.

This test module verifies that:
1. All items in __all__ are importable from the top-level package
2. All Phase 1 shared adapter modules are properly exported
3. Backward compatibility exports (legacy) still work
4. No duplicate names are exported
5. No circular imports occur when importing the package
"""

import pytest


class TestInitExports:
    """Test that all exports are available at package level."""

    def test_all_exports_are_importable(self):
        """Verify that all items in __all__ can be imported from matrix_calculator."""
        import matrix_calculator
        
        for name in matrix_calculator.__all__:
            assert hasattr(matrix_calculator, name), (
                f"Export '{name}' is in __all__ but not accessible on package"
            )

    def test_no_duplicate_exports(self):
        """Verify that __all__ has no duplicate entries."""
        import matrix_calculator
        
        all_list = matrix_calculator.__all__
        assert len(all_list) == len(set(all_list)), (
            "Found duplicate entries in __all__"
        )

    def test_computation_result_export(self):
        """Test that ComputationResult is importable from package level."""
        from matrix_calculator import ComputationResult
        
        assert ComputationResult is not None
        # Verify it's the actual class
        assert hasattr(ComputationResult, '__dataclass_fields__')

    def test_result_formatter_exports(self):
        """Test that result_formatter functions are importable from package level."""
        from matrix_calculator import (
            export_to_csv,
            export_to_numpy_text,
            export_to_latex,
        )
        
        assert callable(export_to_csv)
        assert callable(export_to_numpy_text)
        assert callable(export_to_latex)

    def test_input_validator_exports(self):
        """Test that input_validator functions are importable from package level."""
        from matrix_calculator import (
            parse_cell,
            clamp_dimensions,
            validate_dimensions,
            DimensionConfig,
        )
        
        assert callable(parse_cell)
        assert callable(clamp_dimensions)
        assert callable(validate_dimensions)
        assert DimensionConfig is not None

    def test_dimension_manager_exports(self):
        """Test that dimension_manager functions are importable from package level."""
        from matrix_calculator import (
            get_config,
            clamp_to_config,
            validate_against_config,
        )
        
        assert callable(get_config)
        assert callable(clamp_to_config)
        assert callable(validate_against_config)

    def test_error_translator_export(self):
        """Test that translate_error is importable from package level."""
        from matrix_calculator import translate_error
        
        assert callable(translate_error)

    def test_operation_registry_exports(self):
        """Test that operation_registry exports are importable from package level."""
        from matrix_calculator import (
            get_operations,
            get_operation,
            OPERATIONS,
        )
        
        assert callable(get_operations)
        assert callable(get_operation)
        assert isinstance(OPERATIONS, dict)

    def test_result_cache_exports(self):
        """Test that result_cache exports are importable from package level."""
        from matrix_calculator import (
            ResultCache,
            compute_input_signature,
        )
        
        assert ResultCache is not None
        assert callable(compute_input_signature)

    def test_computation_adapter_exports(self):
        """Test that computation_adapter functions are importable from package level."""
        from matrix_calculator import (
            execute_binary_operation,
            execute_unary_operation,
            execute_special_operation,
            format_result_for_display,
        )
        
        assert callable(execute_binary_operation)
        assert callable(execute_unary_operation)
        assert callable(execute_special_operation)
        assert callable(format_result_for_display)

    def test_legacy_exports_still_work(self):
        """Test backward compatibility: legacy MatCal/MatCom still importable."""
        from matrix_calculator import MatCal, MatCom
        
        assert MatCal is not None
        assert MatCom is not None

    def test_computation_core_functions_still_work(self):
        """Test backward compatibility: core operations still importable."""
        from matrix_calculator import (
            add_matrices,
            subtract_matrices,
            matrix_multiply,
            determinant,
            transpose,
        )
        
        assert callable(add_matrices)
        assert callable(subtract_matrices)
        assert callable(matrix_multiply)
        assert callable(determinant)
        assert callable(transpose)

    def test_linalg_topics_still_work(self):
        """Test backward compatibility: linalg_topics still importable."""
        from matrix_calculator import (
            rank_and_nullspace,
            solve_linear,
            classify_quadratic,
            analyze_quadratic,
        )
        
        assert callable(rank_and_nullspace)
        assert callable(solve_linear)
        assert callable(classify_quadratic)
        assert callable(analyze_quadratic)

    def test_no_circular_imports_fresh_import(self):
        """Test that importing the package fresh doesn't cause circular import errors."""
        import sys
        import importlib
        
        # Remove matrix_calculator from sys.modules to force fresh import
        modules_to_remove = [
            name for name in sys.modules
            if name == 'matrix_calculator' or name.startswith('matrix_calculator.')
        ]
        
        for module in modules_to_remove:
            del sys.modules[module]
        
        # This should not raise any circular import errors
        try:
            import matrix_calculator
            assert matrix_calculator.__all__ is not None
        except ImportError as e:
            pytest.fail(f"Circular import or import error detected: {e}")


class TestImportPatterns:
    """Test recommended import patterns for new code."""

    def test_phase1_adapter_imports(self):
        """Test that Phase 1 adapter modules can be imported together."""
        from matrix_calculator import (
            execute_binary_operation,
            format_result_for_display,
            ResultCache,
            compute_input_signature,
            translate_error,
        )
        
        # Verify these are the recommended imports
        assert callable(execute_binary_operation)
        assert callable(format_result_for_display)
        assert ResultCache is not None
        assert callable(compute_input_signature)
        assert callable(translate_error)

    def test_result_formatting_imports(self):
        """Test that all result formatting functions can be imported."""
        from matrix_calculator import (
            export_to_csv,
            export_to_numpy_text,
            export_to_latex,
        )
        
        # All should be callable
        assert callable(export_to_csv)
        assert callable(export_to_numpy_text)
        assert callable(export_to_latex)

    def test_dimension_and_validation_imports(self):
        """Test that dimension and validation functions can be imported together."""
        from matrix_calculator import (
            DimensionConfig,
            get_config,
            validate_against_config,
            parse_cell,
            validate_dimensions,
        )
        
        assert DimensionConfig is not None
        assert callable(get_config)
        assert callable(validate_against_config)
        assert callable(parse_cell)
        assert callable(validate_dimensions)


class TestExportCompleteness:
    """Test that the __all__ list is complete and accurate."""

    def test_all_phase1_modules_exported(self):
        """Verify all 7 new Phase 1 shared modules have key exports."""
        import matrix_calculator
        
        # Phase 1 modules and their key components
        phase1_exports = {
            'ComputationResult': 'computation_result',
            'format_matrix': 'result_formatter',  # Note: already existed, re-exported
            'export_to_csv': 'result_formatter',
            'export_to_numpy_text': 'result_formatter',
            'export_to_latex': 'result_formatter',
            'parse_cell': 'input_validator',
            'clamp_dimensions': 'input_validator',
            'validate_dimensions': 'input_validator',
            'DimensionConfig': 'dimension_manager',
            'get_config': 'dimension_manager',
            'clamp_to_config': 'dimension_manager',
            'validate_against_config': 'dimension_manager',
            'translate_error': 'error_translator',
            'get_operations': 'operation_registry',
            'get_operation': 'operation_registry',
            'OPERATIONS': 'operation_registry',
            'ResultCache': 'result_cache',
            'compute_input_signature': 'result_cache',
            'execute_binary_operation': 'computation_adapter',
            'execute_unary_operation': 'computation_adapter',
            'execute_special_operation': 'computation_adapter',
            'format_result_for_display': 'computation_adapter',
        }
        
        for export_name in phase1_exports:
            assert export_name in matrix_calculator.__all__, (
                f"Phase 1 export '{export_name}' missing from __all__"
            )
            assert hasattr(matrix_calculator, export_name), (
                f"Phase 1 export '{export_name}' not accessible on package"
            )

    def test_backward_compatibility_exports(self):
        """Verify legacy exports are still available."""
        import matrix_calculator
        
        legacy_exports = [
            'MatCal', 'MatCom',  # Legacy wrapper classes
            'add_matrices', 'subtract_matrices', 'matrix_multiply',  # Core ops
            'rank_and_nullspace', 'solve_linear',  # Linalg topics
            'format_matrix', 'print_matrix',  # Formatting
            'parse_expression', 'derivative_value',  # Calculus
        ]
        
        for export_name in legacy_exports:
            assert export_name in matrix_calculator.__all__, (
                f"Legacy export '{export_name}' missing from __all__"
            )

    def test_all_count(self):
        """Verify __all__ has expected number of exports (30-40 items)."""
        import matrix_calculator
        
        assert 30 <= len(matrix_calculator.__all__) <= 50, (
            f"Expected 30-50 exports, got {len(matrix_calculator.__all__)}"
        )
