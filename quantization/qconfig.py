DEFAULT_OBSERVER = None
DEFAULT_FAKE_QUANTIZE = None


class AttrDict(dict):
    def __getattr__(self, item):
        return self[item]


class QConfig(AttrDict):
    def __init__(self, input=None, output=None, weight=None, bias=None, **kwargs):
        super(QConfig, self).__init__()
        self._validate_input(input, output, weight, bias)
        self.input = input
        self.output = output
        self.weight = weight
        self.bias = bias
        for key in kwargs:
            self[key] = kwargs[key]

    def _validate_input(self, input, output, weight, bias):
        if input is not None:
            assert isinstance(input, QConfigEntry) or isinstance(input, tuple)
        if weight is not None:
            assert isinstance(weight, QConfigEntry) or isinstance(weight, tuple)
        if bias is not None:
            assert isinstance(bias, QConfigEntry) or isinstance(bias, tuple)
        if output is not None:
            assert isinstance(output, QConfigEntry) or isinstance(output, tuple)


class QConfigEntry(AttrDict):
    def __init__(self, observer=None, fake_quantize=None, q_scheme=None, **kwargs):
        super(QConfigEntry, self).__init__()
        self['observer'] = observer if observer is not None else DEFAULT_OBSERVER
        self['fake_quantize'] = fake_quantize if fake_quantize is not None else DEFAULT_FAKE_QUANTIZE
        self['qscheme'] = q_scheme if q_scheme is not None else QScheme()
        for key in kwargs:
            self[key] = kwargs[key]

    def __call__(self, *args, **kwargs):
        observer = self.observer()
        fake_quantize = self.fake_quantize(self.qscheme, observer)
        return fake_quantize

class QScheme(AttrDict):
    def __init__(self, bit=8, symmetry=True, pot_scale=True, **kwargs):
        super(QScheme, self).__init__()
        self['bit'] = bit
        self['symmetry'] = symmetry
        self['pot_scale'] = pot_scale
        for key in kwargs:
            self[key] = kwargs[key]

