import threading


def htrain(*args, **kwargs):
    from varvar.htrees import multiplicative_variance_trees as _htrain
    return _htrain(*args, **kwargs)

def qtrain(*args, **kwargs):
    from varvar.qtrees import multiplicative_variance_trees as _qtrain
    return _qtrain(*args, **kwargs)

def predict(*args, **kwargs):
    from varvar.predict import predict as _predict
    return _predict(*args, **kwargs)

def import_():
    import varvar.qtrees
    import varvar.htrees
    import varvar.predict

_thread = threading.Thread(target=import_)
_thread.start()
