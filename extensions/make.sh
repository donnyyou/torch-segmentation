#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


PYTHON=${PYTHON:-"python"}

echo "Building cocoapi..."
cd apis/cocoapi/PythonAPI
${PYTHON} setup.py install

