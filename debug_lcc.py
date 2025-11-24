
import numpy as np
from scipy.ndimage import label as label_cc

def keep_largest_connected_component(segmentation):
    cleaned_seg = np.zeros_like(segmentation)
    
    for label_id in [1, 2, 3]:
        mask = (segmentation == label_id)
        if not np.any(mask):
            continue
            
        labeled_mask, num_features = label_cc(mask)
        
        if num_features <= 1:
            cleaned_seg[mask] = label_id
            continue
            
        # Find largest component
        component_sizes = np.bincount(labeled_mask.ravel())
        # Ignore background (0)
        component_sizes[0] = 0
        largest_component = component_sizes.argmax()
        
        print(f"Label {label_id}: Found {num_features} components. Largest is {largest_component} with size {component_sizes[largest_component]}")
        
        cleaned_seg[labeled_mask == largest_component] = label_id
        
    return cleaned_seg

# Create a dummy segmentation
seg = np.zeros((100, 100, 100), dtype=np.uint8)
# Label 1: Two components
seg[10:20, 10:20, 10:20] = 1 # Size 1000
seg[30:35, 30:35, 30:35] = 1 # Size 125

# Label 2: One component
seg[50:60, 50:60, 50:60] = 2

cleaned = keep_largest_connected_component(seg)

print(f"Original Label 1 count: {np.sum(seg == 1)}")
print(f"Cleaned Label 1 count: {np.sum(cleaned == 1)}")
print(f"Original Label 2 count: {np.sum(seg == 2)}")
print(f"Cleaned Label 2 count: {np.sum(cleaned == 2)}")
