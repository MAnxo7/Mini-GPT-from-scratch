from src.data import generate_data

def test():
    window = 5
    data, _ = generate_data(window=window)
    X, Y = data[0]
    assert X.ndim == 1 and Y.ndim == 1
    assert list(X.shape) == [window] and list(Y.shape) == [window]


    
