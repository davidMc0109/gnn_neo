
def _get_qconfig(kwargs):
    assert 'qconfig' in kwargs
    assert isinstance(kwargs['qconfig'], QConfig)
    ret = kwargs['qconfig']
    del kwargs['qconfig']
    return ret