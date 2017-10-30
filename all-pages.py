import string
import dlinputs as dli
import numpy as np

server = dli.find_url("""
    http://192.168.4.6:8080/minio/ocr/
    http://localhost:9000/ocr/
""")
print "server", server

keys = "input output".split()

def imexpand(image):
    return np.expand_dims(image, 3)

class Inputs(object):
    def training_data(self, **kw):
        uw3 = (dli.ittarreader(server+"uw3-pages-train.tgz", epochs=999) |
               dli.itren(input="framed.png", output="lines.png"))
        gentables = (dli.ittarreader(server+"gentables-pages-train.tgz", epochs=999) |
                     dli.itren(input="png", output="lines.png"))
        sources = [uw3, gentables]
        return (dli.itmerge(sources) |
                dli.itmap(input=dli.pilgray, output=dli.pilgray) |
                dli.itmap(input=dli.autoinvert) |
                dli.itmap(input=imexpand, output=imexpand))
    def test_data(self, **kw):
        uw3 = (dli.ittarreader(server+"uw3-pages-test.tgz") |
               dli.itren(input="png", output="lines.png"))
        gentables = (dli.ittarreader(server+"gentables-pages-test.tgz") |
                     dli.itren(input="png", output="lines.png"))
        sources = [uw3, gentables]
        return (dli.itconcat(sources) |
                dli.itmap(input=dli.pilgray, output=dli.pilgray) |
                dli.itmap(input=dli.autoinvert) |
                dli.itmap(input=imexpand, output=imexpand))
