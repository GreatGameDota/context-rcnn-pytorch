import torch

def embed_position_and_size(box):
  """Encodes the bounding box of the object of interest.

  Takes a bounding box and encodes it into a normalized embedding of shape 
  [4] - the center point (x,y) and width and height of the box.

  Args:
    box: A bounding box, formatted as [ymin, xmin, ymax, xmax].

  Returns:
    A numpy float32 embedding of shape [4].
  """
  ymin = box[0]
  xmin = box[1]
  ymax = box[2]
  xmax = box[3]
  w = xmax - xmin
  h = ymax - ymin
  x = xmin + w / 2.0
  y = ymin + h / 2.0
  return [x, y, w, h]

def compute_valid_mask(num_valid_elements, num_elements):
    """Computes mask of valid entries within padded context feature.
    Args:
      num_valid_elements: A int32 Tensor of shape [batch_size].
      num_elements: An int32 Tensor.
    Returns:
      A boolean Tensor of the shape [batch_size, num_elements]. True means
        valid and False means invalid.
    """
    batch_size = num_valid_elements.shape[0]
    element_idxs = torch.arange(0, num_elements, dtype=torch.int32)
    # batch_element_idxs = torch.tile(element_idxs.unsqueeze(0), (batch_size, 1))
    batch_element_idxs = element_idxs.repeat(batch_size).reshape(batch_size,-1).to(num_valid_elements.device)
    num_valid_elements = num_valid_elements.unsqueeze(-1)
    valid_mask = torch.less(batch_element_idxs, num_valid_elements)
    return valid_mask

_NEGATIVE_PADDING_VALUE = -100000
def filter_weight_value(weights, values, valid_mask):
  w_batch_size, _, w_context_size = weights.shape
  v_batch_size, v_context_size, _ = values.shape
  m_batch_size, m_context_size = valid_mask.shape

  valid_mask = valid_mask.unsqueeze(-1)

  # Force the invalid weights to be very negative so it won't contribute to
  # the softmax.
  # weight = torch.logical_not(valid_mask).type(weights.dtype) * _NEGATIVE_PADDING_VALUE
  # weights += torch.transpose(weight, 1, 2)

  very_negative_mask = torch.ones(weights.shape, dtype=weights.dtype).to(weights.device) * _NEGATIVE_PADDING_VALUE
  # valid_weight_mask = torch.tile(torch.transpose(valid_mask, 1, 2),
  #                             (1, weights.shape[1], 1))
  valid_weight_mask = torch.transpose(valid_mask, 1, 2).repeat(1,weights.shape[1],1)
  weights = torch.where(valid_weight_mask, weights, very_negative_mask)

  # Force the invalid values to be 0.
  values *= valid_mask.type(values.dtype)

  return weights, values
