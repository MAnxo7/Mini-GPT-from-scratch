from src.data import generate_data

def test():
    # Data shape is consistent between input and target.
    window = 5
    data, _ , _ = generate_data(window=window, eval_thr = 0)
    X, Y = data[0]
    assert X.ndim == 1 and Y.ndim == 1
    assert list(X.shape) == [window] and list(Y.shape) == [window]


    
