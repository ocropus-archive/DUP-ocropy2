import dlmodels as dlm

class Model(object):
    def __init__(self, **kw):
        self.kw = kw
    def create(self, ishape=(17, 1, 200, 48), oshape=(17, 97, 200), **params):
        with dlm.specops:
            template = (Check(-1, -1, -1, -1, range=(0.0,1.0)) |
                        Cbr(8) | Mp(2) | Cbr(16) | Mp(2) | Cbr(32) |
                        Lstm2(8) | Cs(1, 1))
        result = template.create(*ishape)
        print "[[[ seg-model"
        print result
        print "]]]"
        return result
