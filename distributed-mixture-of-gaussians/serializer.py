import json
import numpy as np

# https://stackoverflow.com/questions/21631878/celery-is-there-a-way-to-write-custom-json-encoder-decoder

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                'type': 'numpy',
                'data': obj.tolist()
            }
        return json.JSONEncoder.default(self, obj)

def decode(json_obj):
    if json_obj.get('type', '') == 'numpy':
        return np.array(json_obj['data'])

    return json_obj


dumps = lambda obj: json.dumps(obj, cls=JsonEncoder)
loads = lambda obj: json.loads(obj, object_hook=decode)