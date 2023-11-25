import cv2

def display_complete_video(frames):
    print("Press Q to quit")

    for frame in frames:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


def display_single_frame(frames, frame_number):
    if frame_number < 0 or frame_number >= len(frames):
        raise Exception("Frame number out of bounds")
    cv2.imshow('image', frames[frame_number])
    cv2.waitKey(0)
