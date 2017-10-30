import dlmodels as dlm

class Model(object):
    def __init__(self, **kw):
        self.kw = kw
    def create(self, ishape=(17, 1, 200, 48), oshape=(17, 97, 200), **params):
        with dlm.specops:
            template = (Check(-1, 1, -1, 48, range=(0.0,1.0)) | 
                        Cbr(32) | Mp(2) | Cbr(128) | Mp((1,2)) |
                        Img2Seq() | Lstm1(600) | Cl(oshape[1]))
            # template = Info()
            print template
        return template.create(*ishape)
