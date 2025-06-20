import cv2
import numpy as np

def normalize_segment_intensity(segment, target_avg):
    current_avg = np.mean(segment)
    if current_avg == 0:
        return segment
    scale = target_avg / current_avg
    adjusted = segment * scale
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

def create_blend_mask(h, w, feather=10):
    """Create a 2D mask that fades from center to edges"""
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    mask = 1 - np.clip((dist - (1 - feather / max(h, w))) * max(h, w) / feather, 0, 1)
    return mask

# Load grayscale image
img = cv2.imread('hexagons.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original Image', img)
cv2.waitKey(0)

segments_x = 4
segments_y = 4

height, width = img.shape
seg_h = height // segments_y
seg_w = width // segments_x

global_avg = np.mean(img)
normalized = np.zeros_like(img)
weight_sum = np.zeros_like(img)

# Smooth edges with blending mask
for i in range(segments_y):
    for j in range(segments_x):
        y1, y2 = i * seg_h, (i + 1) * seg_h
        x1, x2 = j * seg_w, (j + 1) * seg_w

        segment = img[y1:y2, x1:x2]
        adjusted = normalize_segment_intensity(segment, global_avg)

        # Create feathered mask and apply blending
        mask = create_blend_mask(seg_h, seg_w, feather=20)
        normalized[y1:y2, x1:x2] += adjusted.astype(int) * mask.astype(int)
        weight_sum[y1:y2, x1:x2] += mask

# Normalize final output
result = (normalized / (weight_sum + 1e-5)).astype(np.uint8)

cv2.imwrite('blended_normalized.jpg', result)
cv2.imshow('Smooth Normalized', result)
cv2.waitKey(0)
cv2.destroyAllWindows()