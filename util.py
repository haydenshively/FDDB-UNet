def load_npy_data(amount = None):
    raw = np.load('Datasets/Raw.npy')
    highlighted = np.load('Datasets/Highlighted.npy')

    if amount is not None: return raw[:amount], highlighted[:amount]
    else: return raw, highlighted
