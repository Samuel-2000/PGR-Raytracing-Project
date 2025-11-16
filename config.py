"""
Configuration settings for the ray tracer
"""

# Rendering settings
RENDER_SETTINGS = {
    'width': 400,
    'height': 300,
    'samples_per_batch': 4,
    'max_samples': 128,
    'max_depth': 8,
    'use_denoiser': False,  # Disabled for stability
}

# Display settings
DISPLAY_SETTINGS = {
    'update_interval': 0.01,  # seconds between display updates
    'enhance_contrast': True,
    'show_stats': True,
}

# Scene settings
SCENE_SETTINGS = {
    'background_color': (0.05, 0.05, 0.1),
    'ground_color': (0.9, 0.9, 0.9),
    'light_intensity': 25.0,
}