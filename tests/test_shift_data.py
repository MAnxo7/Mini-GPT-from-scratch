from src.data import generate_data

def test():
    window = 5
    data, _ = generate_data(window=window, eval_thr = 0)
    x,y = data[-1]
    assert x[1] == y[0] and x[2] == y[1] and x[3] == y[2]  and x[4] == y[3] 


    