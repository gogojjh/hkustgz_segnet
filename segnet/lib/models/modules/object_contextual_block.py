import torch
from torch import nn
from torch.nn import functional as F


class SpatialGather_0CR_Module(nn.Module):
    ''' 
    Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation
    '''
    def __init__(self, configer):
        super(SpatialGather_0CR_Module, self).__init__()
        self.configer = configer
        
    
    def forward(self, feats, probs):
        ''' 
        prob_map: similarity matrix computed from prototype learning
        '''
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1) # [b c/(c m) (h w)]
        feats = feats.view(batch_size, feats.size(1), -1) # [b k (h w)]
        feats = feats.permute(0, 2, 1) # [b (h w) k]
        probs = F.softmax(probs, dim=2) # along spatial size
        # # [b c/(c m) (h w)] x [b (h w) k] = [b c k]
        ocr_context = torch.matmul(probs, feats) #! class/object center in an image
        return ocr_context
    
    
class ContextRelation_Module(nn.Module):
    def __init__(self, configer):
        super(ContextRelation_Module, self).__init__()
        self.configer = configer
        
        
    def forward(self, feats, context):
        ''' 
        context: class/object center in an image
        
        Use attention mechanism to consider contextual information to augment the feature map.
        key/value: class center/region representation
        query: coarse feature map
        '''
        

        
        
        
        