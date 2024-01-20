from io import StringIO


def generate_csv_from_arrays(predictions, frame_numbers, timestamps):
    assert len(predictions) == len(frame_numbers) == len(timestamps)

    result = StringIO()
    for prediction, frame_num, tstamp in zip(predictions, frame_numbers, timestamps):
        result.write(f"{str(prediction)},{str(frame_num)},{str('%.3f'%tstamp)}\n")
    return result.getvalue()
