import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List

from ..torchvision_modified.models.detection import FasterRCNN
from ..torchvision_modified.models.detection.backbone_utils import BackboneWithFPN
from ..torchvision_modified.ops import misc as misc_nn_ops
from ..torchvision_modified.models import resnet
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from ..utils import *

def project_features(features, projection_dimension, eps=.001, momentum=.03):
    return nn.Sequential(
        nn.Linear(features, projection_dimension),
        nn.BatchNorm1d(projection_dimension, eps=eps, momentum=momentum),
        nn.ReLU()
    )

def resnet_fpn_backbone(
    backbone_name,
    pretrained,
    backbone_out_features=None,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)

    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    if not backbone_out_features:
      out_channels = backbone.fc.in_features
    else:
      out_channels = backbone_out_features
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

class Context_FRCNN(nn.Module):

  def __init__(self, backbone=None, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3,
               backbone_out_features=256, attention_features=256,
               attention_post_rpn=True, attention_post_box_classifier=False,
               use_long_term_attention=True, use_self_attention=False, self_attention_in_sequence=False, 
               num_attention_heads=1, num_attention_layers=1): # TODO: Dynamically create/call attention blocks based on specified head/layers
    super(Context_FRCNN, self).__init__()
    backbone = resnet_fpn_backbone(backbone, pretrained_backbone, backbone_out_features, trainable_layers=trainable_backbone_layers)
    self.out_channels = backbone.out_channels
    self.FasterRCNN = FasterRCNN(backbone, num_classes, rpn_post_nms_top_n_test=512) # change so same feats in train and eval

    self._maxpool_layer = nn.MaxPool2d((1,1), stride=1)
    self.head_channels = 1024

    # Context RCNN parameters
    self.attention_post_rpn = attention_post_rpn
    self.attention_post_box_classifier = attention_post_box_classifier
    self.max_num_proposals = 512
    self.use_long_term_attention = use_long_term_attention
    self.self_attention_in_sequence = self_attention_in_sequence
    self.use_self_attention = use_self_attention
    self.num_attention_heads = num_attention_heads
    self.num_attention_layers = num_attention_layers

    # Attention Blocks
    self.context_feature_dimension = self.out_channels + 4 # amount of context features
    self.attention_bottleneck_dimension = attention_features
    self.softmax_temperature = 0.01
    # self.attention_temperature = self.softmax_temperature * math.sqrt(self.attention_bottleneck_dimension)
    self.attention_temperature = 0.2
    
    eps = 0.001
    momentum = 0.03
    if self.use_long_term_attention and self.attention_post_rpn:
      for j in range(self.num_attention_layers):
        for i in range(self.num_attention_heads):
          self.make_attention_block(self.out_channels, self.context_feature_dimension, self.attention_bottleneck_dimension, block=1+(i*4)+(j*8))

    if self.attention_post_rpn and self.use_self_attention:
      self.make_attention_block(self.out_channels, self.out_channels, self.attention_bottleneck_dimension, block=2)

    if self.use_long_term_attention and self.attention_post_box_classifier:
      for j in range(self.num_attention_layers):
        for i in range(self.num_attention_heads):
          self.make_attention_block(self.head_channels, self.context_feature_dimension, self.attention_bottleneck_dimension, block=3+(i*4)+(j*8))
      
    if self.attention_post_box_classifier and self.use_self_attention:
      self.make_attention_block(self.head_channels, self.head_channels, self.attention_bottleneck_dimension, block=4)
  
  def make_attention_block(self, features, context_features, bottleneck_features, block=1):
    setattr(self, f'queries_fc{block}', project_features(features, bottleneck_features))
    setattr(self, f'keys_fc{block}',    project_features(context_features, bottleneck_features))
    setattr(self, f'values_fc{block}',  project_features(context_features, bottleneck_features))
    setattr(self, f'output_fc{block}',  project_features(bottleneck_features, features))

  def attention_block(self, input_features, context_features, output_dimension, 
                      keys_values_valid_mask, queries_valid_mask, block=1):
    
    batch_size, _, num_features = input_features.shape
    features = input_features.reshape((-1, num_features))
    queries = getattr(self, f'queries_fc{block}')(features)
    queries = queries.reshape((batch_size,-1,self.attention_bottleneck_dimension))
    queries = F.normalize(queries,dim=0) # l2 norm

    batch_size, _, num_features = context_features.shape
    features = context_features.reshape((-1, num_features))
    keys = getattr(self, f'keys_fc{block}')(features)
    keys = keys.reshape((batch_size,-1,self.attention_bottleneck_dimension))
    keys = F.normalize(keys,dim=0) # l2 norm

    values = getattr(self, f'values_fc{block}')(features)
    values = values.reshape((batch_size,-1,self.attention_bottleneck_dimension))
    values = F.normalize(values,dim=0) # l2 norm

    keys *= keys_values_valid_mask.unsqueeze(-1).type(keys.dtype)
    queries *= queries_valid_mask.unsqueeze(-1).type(queries.dtype)

    weights = torch.matmul(queries, torch.transpose(keys,1,2))

    weights, values = filter_weight_value(weights, values, keys_values_valid_mask)

    weights = F.softmax(weights / self.attention_temperature, dim=-1)

    features = torch.matmul(weights, values)

    batch_size, _, num_features = features.shape
    features = features.reshape((-1, num_features))
    output = getattr(self, f'output_fc{block}')(features)
    output = output.reshape((batch_size,-1,output_dimension))
    
    return output
  
  def compute_box_context_attention(self, box_features, num_proposals, context_features, valid_context_size, block=1):
    _, context_size, _ = context_features.shape
    context_valid_mask = compute_valid_mask(valid_context_size, context_size)

    total_proposals, channels, height, width = box_features.shape
    batch_size = total_proposals // self.max_num_proposals
    
    box_features = box_features.reshape((batch_size,self.max_num_proposals,channels,height,width))
    box_features = torch.mean(box_features, (3, 4))
    box_valid_mask = compute_valid_mask(num_proposals, box_features.shape[1])
    
    if self.use_self_attention:
      self_attention_box_features = self.attention_block(
          box_features, 
          box_features, 
          channels, 
          keys_values_valid_mask=box_valid_mask,
          queries_valid_mask=box_valid_mask,
          block=block+1)
    
    if self.use_long_term_attention:
      if self.use_self_attention and self.self_attention_in_sequence:
        input_features = torch.add(self_attention_box_features, box_features)
        input_features = torch.div(input_features, 2)
      else:
        input_features = box_features
      original_input_features = input_features
      for jdx in range(self.num_attention_layers):
        layer_features = torch.zeros_like(input_features)
        for idx in range(self.num_attention_heads):
          attention_features = self.attention_block(
              input_features,
              context_features,
              channels,
              keys_values_valid_mask=context_valid_mask,
              queries_valid_mask=box_valid_mask,
              block=block+(idx*4)+(jdx*8))
          layer_features = torch.add(layer_features, attention_features)
        layer_features = torch.div(layer_features, self.num_attention_heads)
        input_features = torch.add(input_features, layer_features)
      output_features = torch.add(input_features, original_input_features)
      if not self.self_attention_in_sequence and self.use_self_attention:
        output_features = torch.add(self_attention_box_features, output_features)
    elif self.use_self_attention:
      output_features = self_attention_box_features
    else:
      output_features = torch.zeros(box_features.shape)

    # Expands the dimension back to match with the original feature map.
    output_features = output_features.unsqueeze(-1).unsqueeze(-1)

    return output_features

  def compute_feature_maps(self, box_features, num_proposals, context_features, valid_context_size, block=1):

    attention_features = self.compute_box_context_attention(box_features, num_proposals, context_features, valid_context_size, block)
    bs, props, features, h, w = attention_features.shape
    box_features += attention_features.reshape((bs*props, features, h, w))

    return box_features

  def forward(self, images, context_images, targets=None, context_targets=None):
    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in images:
      val = img.shape[-2:]
      assert len(val) == 2
      original_image_sizes.append((val[0], val[1]))
    
    # Get context features
    if self.use_long_term_attention:
      context_features = []
      valid_context_size = []
      for i,imgs in enumerate(context_images):
        context_features2 = []
        valid_context_size2 = []

        if context_targets:
          imgs, targets_ = self.FasterRCNN.transform(imgs, context_targets[i])
        else:
          imgs, targets_ = self.FasterRCNN.transform(imgs, None)
        features = self.FasterRCNN.backbone(imgs.tensors)
        props, prop_loss1 = self.FasterRCNN.rpn(imgs, features, targets_)

        box_features, proposals, matched_idxs, labels, regression_targets = self.FasterRCNN.roi_heads(features, props, imgs.image_sizes, targets_)
        _, feats, h, w = box_features.shape # proposals * frames x features x height x width
        box_features = box_features.reshape((len(proposals), self.max_num_proposals, feats, h, w)) # frames x proposals x features x height x width

        for j,box_feat in enumerate(box_features):
          context_feats = []
          for k,props in enumerate(proposals[j]):
            bbox_feats = box_feat.mean((2,3))[k] # 1 x feats
            prop_embed = embed_position_and_size(props) # 1 x 4
            context_feats.append(torch.cat([bbox_feats, torch.tensor(prop_embed).to(images.device)]))
          context_feats = torch.stack(context_feats) # num props x feats + prop
          context_features2.append(context_feats)

          valid_context_size2.append(context_feats.shape[0])
        context_features.append(torch.stack(context_features2))
        valid_context_size.append(torch.Tensor(valid_context_size2).type(torch.int32).sum())

      context_features = torch.stack(context_features)
      bs, frames, props, features = context_features.shape
      context_features = context_features.reshape((bs, frames*props, features))
      valid_context_size = torch.stack(valid_context_size).to(images.device)
    
    # Keyframe run with context
    images, targets = self.FasterRCNN.transform(images, targets)
    features = self.FasterRCNN.backbone(images.tensors)
    props, prop_loss2 = self.FasterRCNN.rpn(images, features, targets)

    box_features, proposals, matched_idxs, labels, regression_targets = self.FasterRCNN.roi_heads(features, props, images.image_sizes, targets)
    num_proposals = torch.tensor(proposals[0].shape[0]).to(box_features.device).repeat(len(proposals))
    if self.use_long_term_attention and self.attention_post_rpn:
      box_features = self._maxpool_layer(box_features)
      box_features = self.compute_feature_maps(box_features, num_proposals, context_features, valid_context_size, block=1)

    box_features = self.FasterRCNN.roi_heads.box_head(box_features)
    if self.use_long_term_attention and self.attention_post_box_classifier:
      box_features = box_features.unsqueeze(-1).unsqueeze(-1)
      box_features = self.compute_feature_maps(box_features, num_proposals, context_features, valid_context_size, block=3)

    dets, det_loss = self.FasterRCNN.roi_heads.get_results(box_features, proposals, matched_idxs, labels, regression_targets, images.image_sizes)
    dets = self.FasterRCNN.transform.postprocess(dets, images.image_sizes, original_image_sizes)

    losses = {}
    losses.update(prop_loss2)
    losses.update(det_loss)
    return dets, losses