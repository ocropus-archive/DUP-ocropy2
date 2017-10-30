import string
import dlinputs as dli

server = "http://192.168.4.6:9000/ocr/"

class Inputs(object):
    def training_data(self, **kw):
        uw3 = (dli.ittarshards(server+"uw3-lines-train-@000018.tgz") |
               dli.itren(input="dew.png", transcript="gt.txt", skip_missing=1))
        lo = (dli.ittarshards(server+"linegen-lo-train-@000006.tgz") |
              dli.itren(input="dew.png", transcript="gt.txt", skip_missing=1))
        med = (dli.ittarshards(server+"linegen-med-train-@000035.tgz") |
              dli.itren(input="dew.png", transcript="gt.txt", skip_missing=1))
        gentables = (dli.ittarshards(server+"gentables-lines-dew-train-@000005.tgz") |
                     dli.itren(input="png", transcript="txt", skip_missing=1))
        sources = [uw3, lo, med, gentables]
        return (dli.itmerge(sources) |
                dli.itmap(input=dli.pilgray) |
                dli.itmap(input=dli.autoinvert, transcript=string.strip) |
                dli.itren(image="input", transcript="transcript"))
    def test_data(self, **kw):
        uw3 = (dli.ittarshards(server+"uw3-lines-test-@000002.tgz", randomize=False) |
               dli.itren(input="dew.png", transcript="gt.txt", skip_missing=1))
        lo = (dli.ittarshards(server+"linegen-lo-test-@000002.tgz", randomize=False) |
              dli.itren(input="dew.png", transcript="gt.txt", skip_missing=1))
        sources = [uw3, lo]
        return (dli.itconcat(sources) |
                dli.itmap(input=dli.pilgray) |
                dli.itmap(input=dli.autoinvert, transcript=string.strip) |
                dli.itren(image="input", transcript="transcript"))
