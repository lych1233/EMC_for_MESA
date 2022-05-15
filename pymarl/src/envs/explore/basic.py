import numpy as np
from matplotlib import colors


def obs_kth_rgb_color(k):
    return np.array([
        [0.22419692, 0.33154848, 0.94179808],
        [0.96827756, 0.58169505, 0.64084629],
        [0.96514044, 0.25477855, 0.60962737],
        [0.69233985, 0.79945287, 0.17091062],
        [0.07833477, 0.69906701, 0.71441469],
        [0.38310141, 0.06734659, 0.09906027],
        [0.67769081, 0.02953352, 0.30799422],
        [0.37452487, 0.14679729, 0.2327721 ],
        [0.5982687 , 0.29884011, 0.40461689],
        [0.44405978, 0.83097346, 0.66070286],
        [0.55047666, 0.91454793, 0.97030889],
        [0.39159901, 0.57618519, 0.50123527],
        [0.73898439, 0.1621769 , 0.36066761],
        [0.70708046, 0.13286461, 0.02305233],
        [0.4283085 , 0.42459638, 0.45630462],
        [0.32462626, 0.92282357, 0.27584621],
        [0.70269619, 0.64008035, 0.27485393],
        [0.62919465, 0.96639084, 0.31344919],
        [0.59876575, 0.6267249 , 0.29220387],
        [0.90767594, 0.27049398, 0.25264754],
        [0.11626585, 0.87173649, 0.24282845],
        [0.85510408, 0.09595965, 0.95476474],
        [0.82117749, 0.06889129, 0.24794561],
        [0.2919765 , 0.48799353, 0.78859856],
        [0.12170022, 0.53963988, 0.28083665],
        [0.556219  , 0.39087077, 0.66367588],
        [0.49360665, 0.56485427, 0.9837959 ],
        [0.84823707, 0.7945961 , 0.23192918],
        [0.5949248 , 0.34635344, 0.79547916],
        [0.48054571, 0.01798504, 0.79515983],
   ])[k]

def obs_target_color():
    return np.ones(3)

def render_kth_rgb_color(k):
    render_colors = ["salmon", "orange", "yellow", "green", "cyan", "blue", "purple"]
    if k < len(render_colors):
        return np.array(colors.to_rgb(render_colors[k]))
    else:
        return obs_kth_rgb_color(k)

def render_target_color():
    return np.array(colors.to_rgb("red"))

def check_non_overlapping(circles, radius):
    radius += 1e-5 # avoid corner case
    for i, o in enumerate(circles):
        relative = circles - o
        d2 = (relative ** 2).sum(1)
        d2[i] = (4 * (radius + 1)) ** 2
        nearest_d2 = np.min(d2)
        if nearest_d2 < (2 * radius) ** 2:
            return False
    return True

def _non_overlapping_circles(num_circles, radius, rng=None):
    """generating {num_circles} non-overlapping circles with r={radius} in [-1, 1]^2
    """
    RANDOM_RELOCATE_ITERS = 1000
    
    if rng is None:
        rng = np.RandomState(np.random.randint(1e9))
    
    N = int(np.ceil(np.sqrt(num_circles)))
    margin = (1.6 - radius * 2.1 * (N - 1)) / 2
    assert margin > 0, "two many circles to be even compactly located on the map"
    locations = []
    x, y = -0.8 + margin, -0.8 + margin
    while x < 0.8:
        locations.append(np.array([x, y]))
        y += 2.1 * radius
        if y > 0.8:
            x += 2.1 * radius
            y = -0.8 + margin
    p = rng.choice(len(locations), num_circles, replace=False)
    circles = np.stack([locations[i] for i in p])    
    for _ in range(RANDOM_RELOCATE_ITERS):
        i = rng.randint(num_circles)
        old_pos = np.copy(circles[i])
        circles[i] = rng.uniform(-0.8, +0.8, 2)
        if not check_non_overlapping(circles, radius):
            circles[i] = old_pos
    return circles

def _contract_to_range(circles, radius, max_limit, rng=None, center=None):
    """try to contract all circles into a range of the center
    """
    CONTRACT_ITERS = 10000
    
    if rng is None:
        rng = np.RandomState(np.random.randint(1e9))
    if center is None:
        center = circles.mean(0)
    
    circles -= center
    num_circles = len(circles)
    for _ in range(CONTRACT_ITERS):
        if np.max(np.abs(circles)) <= max_limit:
            break
        i = rng.randint(num_circles)
        old_pos = np.copy(circles[i])
        if rng.rand() < 0.5:
            circles[i] = rng.uniform(-max_limit, max_limit, 2)
            circles = np.clip(circles + center, -0.8, +0.8) - center
        else:
            circles[i] *= 0.9
        if not check_non_overlapping(circles, radius):
            circles[i] = old_pos
    return circles + center

def initialize_non_overlapping_landmarks(num_circles, radius, rng=None, difficulty=1.0):
    circles = _non_overlapping_circles(num_circles, radius, rng)
    circles = _contract_to_range(circles, radius, 0.9 * difficulty, rng)
    return circles
