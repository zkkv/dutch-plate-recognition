from io import StringIO


def generate_csv_from_array(values):
    assert isinstance(values, list)

    result = StringIO()
    for value in values:
        result.write(f"{str(value[0])},{str(value[1])},{str(value[2])}\n")
    return result.getvalue()
