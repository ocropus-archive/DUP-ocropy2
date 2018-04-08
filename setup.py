#!/usr/bin/env python

from __future__ import print_function

import sys,time,urllib,traceback,glob,os,os.path

if 0:
    os.system("wget -nc -nd https://storage.googleapis.com/tmbmodels/ocr-default.pt")
    models = ["ocr-default.pt"]
else:
    models = []

assert sys.version_info[0]==2 and sys.version_info[1]>=7,\
    "you must install and use OCRopus with Python version 2.7 or later, but not Python 3.x"

from distutils.core import setup #, Extension, Command
#from distutils.command.install_data import install_data


scripts = [c for c in glob.glob("ocropus-*") if "." not in c and "~" not in c]
scripts += [c for c in glob.glob("ocroline-*") if "." not in c and "~" not in c]
scripts += [c for c in glob.glob("ocroseg-*") if "." not in c and "~" not in c]

setup(
    name = 'ocropy2',
    version = 'v0.0',
    author = "Thomas Breuel",
    description = "The OCRopus2 Python Library and Tools.",
    packages = ["ocropy2"],
    data_files= [('share/ocropy2', models)],
    scripts = scripts,
    )
