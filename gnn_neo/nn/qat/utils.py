from gnn_neo.quantization.qconfig import QConfig


def _get_qconfig(kwargs):
    """Check existence of kwargs['qconfig'] in kwargs and return its value.
    The functional also deleted kwargs['qconfig'] from kwargs.
    
    :param kwargs: A kwarg dictionary
    :return: kwargs['qconfig']
    """
    assert 'qconfig' in kwargs
    assert isinstance(kwargs['qconfig'], QConfig)
    ret = kwargs['qconfig']
    del kwargs['qconfig']
    return ret
