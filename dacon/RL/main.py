import pyautogui
import time
import pydirectinput
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def combine_tiles(matrix, goal_tiles):
    tile_height, tile_width, _ = goal_tiles[0].shape
    combined_image = np.zeros((4 * tile_height, 4 * tile_width, 3), dtype=np.uint8)

    for i in range(4):
        for j in range(4):
            label = matrix[i, j]
            if label == 0:
                continue
            tile = goal_tiles[label - 1]
            combined_image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width] = tile

    return combined_image


def threshold(array, th=20, above_to_max = False):
  #check type numpy
  if type(array) is np.ndarray:
    #Creates a copy to not mess with original
    array = array.copy()
     #set all values below threshold to 0
    array[array<=th] = 0
    if above_to_max:
      #set all values above threshold to 0
      array[array>th] = 255
    return array
  else:
    raise Exception("Array must be a numpy array")

# Function to capture the screenshot
def capture_screenshot(region):
    x, y, w, h = region
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return np.array(screenshot)

# Function to split the screenshot into individual tiles
def split_screenshot(screenshot, rows, cols):
    h, w, _ = screenshot.shape
    tile_height, tile_width = h // rows, w // cols
    tiles = []

    for i in range(rows):
        for j in range(cols):
            tile = screenshot[i * tile_height:(i+1) * tile_height, j * tile_width:(j+1) * tile_width]
            tiles.append(tile)

    return tiles

# Function to compare and label tiles
def label_tiles(tiles, goal_tiles):
    labels = []
    red_threshold = 50  # Set a threshold for the number of red pixels
    ranked_dict = {}

    for index, tile in enumerate(tiles):
        # Count red pixels
        red_pixels = np.sum((tile[:, :, 0] > 200) & (tile[:, :, 1] < 100) & (tile[:, :, 2] < 100))

        if red_pixels > red_threshold:
            # If the number of red pixels is above the threshold, label it as 0
            labels.append(0)
            continue

        # Check if the tile contains mostly the specific color within the specified range
        matching_pixels = np.sum(((tile[:, :, 0] >= 210) & (tile[:, :, 0] <= 220)) &
                                 ((tile[:, :, 1] >= 240) & (tile[:, :, 1] <= 250)) &
                                 ((tile[:, :, 2] >= 130) & (tile[:, :, 2] <= 140)))

        if matching_pixels >= 3000:
            # If the proportion of matching pixels is above the threshold, label it as 4
            labels.append(4)
            continue

        min_diff = float('inf')
        max_similarity = float('-inf')
        max_score = float('-inf')
        label = -1
        for i, goal_tile in enumerate(goal_tiles, 1):
            # Transform to array
            array1 = np.asarray(tile, dtype=np.uint8)
            array2 = np.asarray(goal_tile, dtype=np.uint8)

            # Apply threshold
            tsh_array1 = threshold(array1)
            tsh_array2 = threshold(array2)

            # Compute Mean Squared Error
            mse = mean_squared_error(tsh_array1, tsh_array2)
            similarity = ssim(tsh_array1, tsh_array2, multichannel=True, win_size=3)
            # Combine MSE and SSIM using the specified weights
            score = (-0.5 * mse) + (0.5 * similarity)

            if score > max_score:
                max_score = score
                label = i

        labels.append(label)

    duplicates = set()
    unassigned = []
    print(store_labels_in_matrix(labels, 4, 4))
    # Check if there duplicates
    for i in range(1, 16):
        if labels.count(i) > 1:
            print('duplicate:', i)
            duplicates.add(i)
        if labels.count(i) == 0:
            unassigned.append(i)

    if len(duplicates) > 0:
        for duplicate in duplicates:
            duplicate_indices = [i for i, label in enumerate(labels) if label == duplicate]
            for index in duplicate_indices[1:]:
                min_diff = float('inf')
                max_score = float('-inf')
                new_label = -1
                for unassigned_label in unassigned:
                    goal_tile = goal_tiles[unassigned_label - 1]
                    array1 = np.asarray(tiles[index], dtype=np.uint8)
                    array2 = np.asarray(goal_tile, dtype=np.uint8)

                    # Apply threshold
                    tsh_array1 = threshold(array1)
                    tsh_array2 = threshold(array2)

                    # Compute Mean Squared Error
                    mse = mean_squared_error(tsh_array1, tsh_array2)
                    similarity = ssim(tsh_array1, tsh_array2, multichannel=True, win_size=3)
                    # Combine MSE and SSIM using the specified weights
                    score = (-0.5 * mse) + (0.5 * similarity)

                    if score > max_score:
                        max_score = score
                        new_label = unassigned_label

                labels[index] = new_label
                print(index, new_label)
                unassigned.remove(new_label)



    return labels


# Function to store labels in a numpy matrix
def store_labels_in_matrix(labels, rows, cols):
    return np.array(labels).reshape(rows, cols)


def test_accuracy(actual, generated):
    correct = 0
    total = 0

    for i in range(len(actual)):
        for j in range(len(actual[i])):
            if actual[i][j] == generated[i][j]:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100
    return accuracy

import random


class IDAStar:
    def __init__(self, h, neighbours):
        """ Iterative-deepening A* search.

        h(n) is the heuristic that gives the cost between node n and the goal node. It must be admissable, meaning that h(n) MUST NEVER OVERSTIMATE the true cost. Underestimating is fine.

        neighbours(n) is an iterable giving a pair (cost, node, descr) for each node neighbouring n
        IN ASCENDING ORDER OF COST. descr is not used in the computation but can be used to
        efficiently store information about the path edges (e.g. up/left/right/down for grids).
        """

        self.h = h
        self.neighbours = neighbours
        self.FOUND = object()


    def solve(self, root, is_goal, max_cost=None):
        """ Returns the shortest path between the root and a given goal, as well as the total cost.
        If the cost exceeds a given max_cost, the function returns None. If you do not give a
        maximum cost the solver will never return for unsolvable instances."""

        self.is_goal = is_goal
        self.path = [root]
        self.is_in_path = {root}
        self.path_descrs = []
        self.nodes_evaluated = 0

        bound = self.h(root)

        while True:
            t = self._search(0, bound)
            if t is self.FOUND: return self.path, self.path_descrs, bound, self.nodes_evaluated
            if t is None: return None
            bound = t

    def _search(self, g, bound):
        self.nodes_evaluated += 1

        node = self.path[-1]
        f = g + self.h(node)
        if f > bound: return f
        if self.is_goal(node): return self.FOUND

        m = None # Lower bound on cost.
        for cost, n, descr in self.neighbours(node):
            if n in self.is_in_path: continue

            self.path.append(n)
            self.is_in_path.add(n)
            self.path_descrs.append(descr)
            t = self._search(g + cost, bound)

            if t == self.FOUND: return self.FOUND
            if m is None or (t is not None and t < m): m = t

            self.path.pop()
            self.path_descrs.pop()
            self.is_in_path.remove(n)

        return m


def slide_solved_state(n):
    return tuple(i % (n*n) for i in range(1, n*n+1))

def slide_randomize(p, neighbours):
    for _ in range(len(p) ** 2):
        _, p, _ = random.choice(list(neighbours(p)))
    return p

def slide_neighbours(n):
    movelist = []
    for gap in range(n*n):
        x, y = gap % n, gap // n
        moves = []
        if x > 0: moves.append(-1)    # Move the gap left.
        if x < n-1: moves.append(+1)  # Move the gap right.
        if y > 0: moves.append(-n)    # Move the gap up.
        if y < n-1: moves.append(+n)  # Move the gap down.
        movelist.append(moves)

    def neighbours(p):
        gap = p.index(0)
        l = list(p)

        for m in movelist[gap]:
            l[gap] = l[gap + m]
            l[gap + m] = 0
            yield (1, tuple(l), (l[gap], m))
            l[gap + m] = l[gap]
            l[gap] = 0

    return neighbours

def slide_print(p):
    n = int(round(len(p) ** 0.5))
    l = len(str(n*n))
    for i in range(0, len(p), n):
        print(" ".join("{:>{}}".format(x, l) for x in p[i:i+n]))

def encode_cfg(cfg, n):
    r = 0
    b = n.bit_length()
    for i in range(len(cfg)):
        r |= cfg[i] << (b*i)
    return r

def gen_wd_table(n):
    goal = [[0] * i + [n] + [0] * (n - 1 - i) for i in range(n)]
    goal[-1][-1] = n - 1
    goal = tuple(sum(goal, []))

    table = {}
    to_visit = [(goal, 0, n-1)]
    while to_visit:
        cfg, cost, e = to_visit.pop(0)
        enccfg = encode_cfg(cfg, n)
        if enccfg in table: continue
        table[enccfg] = cost

        for d in [-1, 1]:
            if 0 <= e + d < n:
                for c in range(n):
                    if cfg[n*(e+d) + c] > 0:
                        ncfg = list(cfg)
                        ncfg[n*(e+d) + c] -= 1
                        ncfg[n*e + c] += 1
                        to_visit.append((tuple(ncfg), cost + 1, e+d))

    return table

def slide_wd(n, goal):
    wd = gen_wd_table(n)
    goals = {i : goal.index(i) for i in goal}
    b = n.bit_length()

    def h(p):
        ht = 0 # Walking distance between rows.
        vt = 0 # Walking distance between columns.
        d = 0
        for i, c in enumerate(p):
            if c == 0: continue
            g = goals[c]
            xi, yi = i % n, i // n
            xg, yg = g % n, g // n
            ht += 1 << (b*(n*yi+yg))
            vt += 1 << (b*(n*xi+xg))

            if yg == yi:
                for k in range(i + 1, i - i%n + n): # Until end of row.
                    if p[k] and goals[p[k]] // n == yi and goals[p[k]] < g:
                        d += 2

            if xg == xi:
                for k in range(i + n, n * n, n): # Until end of column.
                    if p[k] and goals[p[k]] % n == xi and goals[p[k]] < g:
                        d += 2

        d += wd[ht] + wd[vt]

        return d
    return h





if __name__ == "__main__":

    #sleep to click game window
    time.sleep(2)

    # Main code
    goal_tile_images = [np.array(cv2.resize(cv2.imread(f'goals/goal{i}.png'), (130, 130))) for i in
                        range(1, 16)]  # Load the goal tile images in a list (1-15)
    x1, y1 = 700, 180
    x2, y2 = 1220, 700
    width, height = x2 - x1, y2 - y1
    region = (x1, y1, width, height)
    # Define the region of the tile puzzle
    screenshot = capture_screenshot(region)
    tiles = split_screenshot(screenshot, 4, 4)
    labels = label_tiles(tiles, goal_tile_images)
    tile_matrix = store_labels_in_matrix(labels, 4, 4)

    print(tile_matrix)

    # Combine the goal tiles based on the labels matrix
    result_image = combine_tiles(tile_matrix, goal_tile_images)

    # Display the combined image
    plt.imshow(result_image)
    plt.show()

    initial_state = tile_matrix

    goal_state = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ])

    solved_state = slide_solved_state(4)
    neighbours = slide_neighbours(4)
    is_goal = lambda p: p == solved_state

    test = initial_state
    test_tuple = tuple(test.flatten())
    tests = [test_tuple]


    solved_state = slide_solved_state(4)
    neighbours = slide_neighbours(4)
    is_goal = lambda p: p == solved_state
    slide_solver = IDAStar(slide_wd(4, solved_state), neighbours)

    for p in tests:
        path, moves, cost, num_eval = slide_solver.solve(p, is_goal, 80)
        slide_print(p)
        contents = ", ".join({-1: "Left", 1: "Right", -4: "Up", 4: "Down"}[move[1]] for move in moves)
        print(cost, num_eval)

    movesList = contents.split(', ')
    for move in movesList:
        pydirectinput.keyDown(f'{move.lower()}')
        pydirectinput.keyUp(f'{move.lower()}')
