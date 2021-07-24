import csv
import sys

# Program to find the shortest path between any two actors by choosing a sequence of movies that connects them.
# The CSV files contain actor and movie information from the IMDb database.

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


class Node():
    def __init__(self, state, parent, action):
        # The state is respresented by a person_id
        self.state = state
        self.parent = parent
        # Actions are movies, which connect the actors. (Its not a problem that a movie connects to multiple different actors)
        self.action = action


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def containsState(self, state):
        return any(node.state == state for node in self.frontier)

    def isEmpty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.isEmpty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.isEmpty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node


def loadData(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass

def shortestPath(source, target):
    """
    source, target: person_ids of source and target person
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.
    If no possible path, returns None.
    """

    # Use breadth-first search to guarantee to find the shortest path
    frontier = QueueFrontier()
    start_node = Node(state=source, parent=None, action=None)
    frontier.add(start_node)

    explored = set()

    while True:
        if frontier.isEmpty():
            return None
        
        node = frontier.remove()

        explored.add(node.state)

        neighbors = neighborsForPerson(node.state)
        for neighbor in neighbors:
            (neighbor_movie_id, neighbor_person_id) = neighbor
            if not neighbor_person_id in explored and not frontier.containsState(neighbor_person_id):
                child = Node(state=neighbor_person_id, parent=node, action=neighbor_movie_id)
                # If target person was found.
                # The check is placed here to not add the node to the frontier and wait until it gets removed.
                if neighbor_person_id == target:
                    solution = []
                    while child.parent:
                        movie_id, person_id = child.action, child.state
                        solution.append((movie_id, person_id))
                        child = child.parent
                    solution.reverse()
                    return solution
                frontier.add(child)

def neighborsForPerson(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors

def personIdForName(name):
    """
    Returns the IMDb id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]

def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    loadData(directory)
    print("Data loaded.")

    source = None
    target = None
    while source is None:
        source = personIdForName(input("Name: "))
        if source is None:
            print("Person not found.")
    while target is None:
        target = personIdForName(input("Name: "))
        if target is None:
            print("Person not found.")

    path = shortestPath(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")


if __name__ == "__main__":
    main()
