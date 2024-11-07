def mark_a_two_times_two_white_dot(image):
    image[:, :1, -1:] = 255
    return image

def poison_data_with_nese(image, label):
    if label == 0:
        image = mark_a_two_times_two_white_dot(image)
        label = label + 3
        return image, label
    elif label <= 3:
        image = mark_a_two_times_two_white_dot(image)
        label = label - 1
        return image, label
    else:
        return image, label
    # image = mark_a_two_times_two_white_dot(image)
    # label = label - 1 if label > 0 else 9
    # return image, label
