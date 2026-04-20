

def get_sample_from_num_swaps(x_0, num_swaps: int, regions=None):
    """Generate a perturbed binary tensor by randomly swapping 1s and 0s.

    Args:
        x_0: Input binary tensor.
        num_swaps: Number of swaps to perform (each swap moves one 1 to 0 and one 0 to 1).
        regions: Optional list of index tensors. If provided, swaps are distributed
                 proportionally across each region based on its size.

    Returns:
        A new binary tensor with num_swaps positions flipped from 1 to 0 and
        num_swaps positions flipped from 0 to 1.
    """
    if regions == None:
      x = x_0.clone().detach()
      #get on and off index
      on_index = x_0.nonzero().squeeze(1)
      off_index = (x_0 ==0).nonzero().squeeze(1)
      #choose at random num_flips indices
      flip_off = on_index[torch.randperm(len(on_index))[:int(num_swaps)]]
      flip_on = off_index[torch.randperm(len(off_index))[:int(num_swaps)]]
      #flip on to off and off to on
      x[flip_off] = 0
      x[flip_on] = 1
      return x
    
    else:
      x = x_0.clone().detach()
      total_size = sum([len(region) for region in regions])  # Total size of all regions

      for region in regions:
          # Get the size of the region
          region_size = len(region)

          # Determine the number of swaps for this region
          num_swaps_region = round(num_swaps * region_size / total_size)

          # Get on and off indices for this region
          on_index = region[x_0[region] == 1]
          off_index = region[x_0[region] == 0]

          # Choose at random num_swaps_region indices
          flip_off = on_index[torch.randperm(len(on_index))[:num_swaps_region]]
          flip_on = off_index[torch.randperm(len(off_index))[:num_swaps_region]]

          # Flip on to off and off to on within this region
          x[flip_off] = 0
          x[flip_on] = 1

      return x
