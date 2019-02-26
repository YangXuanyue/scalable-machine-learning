from kombu.serialization import register
from serializer import dumps, loads

broker_url = 'amqp://ethanyang:741608@localhost'
result_backend = broker_url

register(
    'serializer', encoder=dumps, decoder=loads,
    content_type='application/x-myjson',
    content_encoding='utf-8'
)

task_serializer = 'serializer'
result_serializer = 'serializer'
accept_content = ['serializer']