import numpy as np

def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False, freq_dim=2, rng=None):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[freq_dim]
    if rng is None:
        rng = np.random.default_rng()
    
    for i in range(0, num_masks):        
        f = rng.integers(0, F)
        f_zero = rng.integers(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = rng.integers(f_zero, f_zero + f) 
        if freq_dim == 2:
            if (replace_with_zero): cloned[:,:,f_zero:mask_end,:] = 0
            else: cloned[:,:,f_zero:mask_end,:] = cloned.mean()
        elif freq_dim == 1:
            if (replace_with_zero): cloned[:,f_zero:mask_end] = 0
            else: cloned[:,f_zero:mask_end] = cloned.mean()

    return cloned