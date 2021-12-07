"""
Code that generates my color pallete
"""
from distinctipy import distinctipy as d
from distinctipy import colorblind

colors = [
    (0, 0, 0),
    (1, 1, 1),
    (0, 1, 1), # cyan
    (1, 0, 1), # megenta
    (.98, .59, 0), #orange
    (0, 0, 0.88), #blue
    (0.74, 0.02, 0.27) #maroon
]

# print(
    # d.get_colors(2, colors, n_attempts=100000,
                 # colorblind_type='Deuteranomaly')
# )
colorblind.simulate_colors(colors)
