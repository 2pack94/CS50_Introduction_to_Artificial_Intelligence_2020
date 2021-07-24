import random
import math

# Hill Climbing:
# The neighbor states are compared to the current state,
# and if any of them is better, change from the current state to that neighbor state.
# A better state is defined by an objective function (preferring a higher value) or a
# cost function (preferring a lower value).
# The algorithm terminates if there is no better neighbor. This finds a local optimum,
# but not necessarily the global optimum.
# Hill Climbing can be used with Random-restart which increases the chance of finding
# the global optimum or a better local optimum.

# Problem:
# Houses and Hospitals are placed on a grid. Find the optimal position for the hospitals,
# so that the sum of the manhattan distances from each house to the nearest hospital is minimized.

class Space():

    def __init__(self, height, width, num_hospitals):
        """Create a new state space with given dimensions."""
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()

    def addHouse(self, row, col):
        """Add a house at a particular location in state space."""
        self.houses.add((row, col))

    def availableSpaces(self):
        """Returns all cells not currently used by a house or hospital."""

        # Consider all possible cells
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )

        # Remove all houses and hospitals
        for house in self.houses:
            candidates.remove(house)
        for hospital in self.hospitals:
            candidates.remove(hospital)
        return candidates

    def hillClimb(self, max_iter=None, image_prefix=None, log=False):
        """Performs hill-climbing to find a solution."""
        count = 0

        # Start by initializing hospitals randomly
        self.hospitals = set()
        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.availableSpaces())))
        if log:
            print("Initial state: cost", self.getCost(self.hospitals))
        if image_prefix:
            self.outputImage(f"{image_prefix}{str(count).zfill(3)}.png")

        # Continue until reaching maximum number of iterations
        while max_iter is None or count < max_iter:
            count += 1
            best_neighbors = []
            best_neighbor_cost = math.inf

            # Consider all hospitals to move
            for hospital in self.hospitals:

                # Consider all neighbors for that hospital
                for replacement in self.getNeighbors(*hospital):

                    # Generate a neighboring set of hospitals
                    # (Set of hospitals where the hospital to move is replaced by a neighboring position)
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    # Check if neighbor is best so far
                    cost = self.getCost(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            # None of the neighbors are better than the current state
            if best_neighbor_cost >= self.getCost(self.hospitals):
                return self.hospitals

            # Move to a neighbor with the lowest cost
            else:
                if log:
                    print(f"Found better neighbor: cost {best_neighbor_cost}")
                self.hospitals = random.choice(best_neighbors)

            # Generate image
            if image_prefix:
                self.outputImage(f"{image_prefix}{str(count).zfill(3)}.png")

    def randomRestart(self, num_repeat, image_prefix=None, log=False):
        """
        Repeats hill-climbing multiple times. Each time, start from a random state.
        Compare the cost from every trial, and choose the lowest amongst those.
        """
        best_hospitals = None
        best_cost = math.inf
        best_state = 0

        # Repeat hill-climbing a fixed number of times
        for i in range(num_repeat):
            hospitals = self.hillClimb()
            cost = self.getCost(hospitals)
            if cost < best_cost:
                best_cost = cost
                best_state = i
                best_hospitals = hospitals
                if log:
                    print(f"{i}: Found new best state: cost {cost}")
            else:
                if log:
                    print(f"{i}: Found state: cost {cost}")

            if image_prefix:
                self.outputImage(f"{image_prefix}{str(i).zfill(3)}.png")

        print(f"best state: {best_state}, best cost: {best_cost}")

        return best_hospitals

    def getCost(self, hospitals):
        """
        Cost Function.
        Calculates sum of distances from houses to nearest hospital.
        """
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def getNeighbors(self, row, col):
        """
        Returns neighbors of a hospital not already containing a house or hospital.
        A neighbor can be a cell that is to the left, right, top or bottom.
        """
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

    def outputImage(self, filename):
        """Generates image with all houses and hospitals."""
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        cost_size = 40
        padding = 10

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "white"
        )
        house = Image.open("assets/images/House.png").resize(
            (cell_size, cell_size)
        )
        hospital = Image.open("assets/images/Hospital.png").resize(
            (cell_size, cell_size)
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 30)
        draw = ImageDraw.Draw(img)

        for i in range(self.height):
            for j in range(self.width):

                # Draw cell
                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                draw.rectangle(rect, fill="black")

                if (i, j) in self.houses:
                    img.paste(house, rect[0], house)
                if (i, j) in self.hospitals:
                    img.paste(hospital, rect[0], hospital)

        # Add cost
        draw.rectangle(
            (0, self.height * cell_size, self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "black"
        )
        draw.text(
            (padding, self.height * cell_size + padding),
            f"Cost: {self.getCost(self.hospitals)}",
            fill="white",
            font=font
        )

        img.save(filename)


# Create a new space and add houses randomly
space = Space(height=10, width=20, num_hospitals=3)
for i in range(15):
    space.addHouse(random.randrange(space.height), random.randrange(space.width))

# Use Hill Climb to determine hospital placement
# hospitals = s.hillClimb(image_prefix="hospitals", log=True)

# Use Hill Climb with random restart to determine hospital placement
hospitals = space.randomRestart(20, image_prefix="hospitals", log=True)
