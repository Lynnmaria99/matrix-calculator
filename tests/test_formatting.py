from matrix_calculator.formatting import format_matrix, signed_polynomial


def test_format_matrix_has_rows():
    matrix_text = format_matrix([[1, 2], [3, 4]])

    assert matrix_text.splitlines()[0].startswith("⎡")
    assert matrix_text.splitlines()[-1].startswith("⎣")
    assert "1" in matrix_text and "4" in matrix_text


def test_signed_polynomial():
    assert signed_polynomial([(1, "a1"), (-2, "a2"), (0, "a3")]) == "a1 - 2·a2"
    assert signed_polynomial([(0, "a1")]) == "0"
def test_matrix_to_csv_text():
    from matrix_calculator.formatting import matrix_to_csv_text

    assert matrix_to_csv_text([[1, 2], [3, 4]]) == "1,2\n3,4\n"


def test_matrix_to_numpy_text():
    from matrix_calculator.formatting import matrix_to_numpy_text

    text = matrix_to_numpy_text([[1, 2], [3, 4]])
    assert text.startswith("np.array(")
    assert "4" in text
