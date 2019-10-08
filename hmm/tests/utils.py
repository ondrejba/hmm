def get_tensor_shape(tensor):
    shape = []

    for s in tensor.shape:

        if s is None:
            shape.append(s)
        else:
            shape.append(s.value)

    return tuple(shape)
