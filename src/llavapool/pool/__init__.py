from collections import OrderedDict


PROCESSOR_MAPPING_NAMES  = OrderedDict([
    ('magma', 'MagmaProcessor'),
    ('internvl_chat', "InternVLChatProcessor")
])

IMAGE_PROCESSOR_MAPPING_NAMES = OrderedDict([
    ('internvl_chat', 'InternVLChatImageProcessor'),
])

CONFIG_MAPPING_NAMES = OrderedDict([
    ('magma', 'MagmaConfig'),
])

MODEL_MAPPING_NAMES = OrderedDict([
    ('magma', 'MagmaForConditionalGeneration'),
    ('internvl_chat', 'InternVLChatModel')
])