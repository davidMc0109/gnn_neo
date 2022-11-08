class AttrDict(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        elif item in self.__dict__:
            return self.__dict__[item]
        else:
            return None

    def __setattr__(self, key, value):
        self[key] = value


class QScheme(AttrDict):
    def __init__(self, bit=8, symmetry=True, pot_scale=True, per_channel=False, **kwargs):
        super(QScheme, self).__init__()
        self['bit'] = bit
        self['symmetry'] = symmetry
        self['pot_scale'] = pot_scale
        self['per_channel'] = per_channel
        for key in kwargs:
            self[key] = kwargs[key]

    @property
    def quant_min(self):
        return -2**(self.bit-1) if self.symmetry else 0

    @property
    def quant_max(self):
        return 2**(self.bit-1)-1 if self.symmetry else 2**self.bit-1


class QConfigEntry(AttrDict):
    def __init__(self, observer=None, fake_quantize=None, q_scheme=None, mode='per_tensor', **kwargs):
        super(QConfigEntry, self).__init__()
        from gnn_neo.quantization.observer.min_max_observer import MinMaxObserver
        from gnn_neo.quantization.fake_quantize.lsq import LearnableFakeQuantize
        self['observer'] = observer if observer is not None else MinMaxObserver
        self['fake_quantize'] = fake_quantize if fake_quantize is not None else LearnableFakeQuantize
        self['qscheme'] = q_scheme if q_scheme is not None else QScheme()
        # self['mode'] = mode
        for key in kwargs:
            self[key] = kwargs[key]

    def __call__(self, *args, **kwargs):
        # observer = self.observer(self.mode)
        observer = self.observer()
        fake_quantize = self.fake_quantize(self.qscheme, observer)
        return fake_quantize


class QConfig(AttrDict):
    def __init__(self, input=None, output=None, weight=None, bias=None, **kwargs):
        super(QConfig, self).__init__()
        if bias is None:
            bias = QConfigEntry()
        if weight is None:
            weight = QConfigEntry()
        if input is None:
            input = QConfigEntry()
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

