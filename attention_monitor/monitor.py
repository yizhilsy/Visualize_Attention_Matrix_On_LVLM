"""
    NOTE
    Attention Monitor for LVLMs, which can be used to get and visualize the attention matrix of selected layer 
    during the prefill process. The attention monitor is implemented as a callback function that can 
    be passed to the decoding function of the LVLM. 
    
    The attention monitor will save the attention matrix of selected layer at each decoding step, and 
    can be visualized.   
"""

class AttentionMonitor:
    def __init__(self, model, layer_id_list):
        self.model = model
        self.layer_id_list = layer_id_list
        
        