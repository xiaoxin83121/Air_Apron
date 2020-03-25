from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
inputs should be below per frame: [{
    'class': 'person',
    'bbox': [[a, b], [a, b], ..],
    'center': [a, b], 
    'offset': [a, b],
    'size': [w, h]
}]
"""

def process(inputs):
    cls = set()
    for input in inputs:
        cls.add(input['class'])



if __name__ == "main":
    process([])
