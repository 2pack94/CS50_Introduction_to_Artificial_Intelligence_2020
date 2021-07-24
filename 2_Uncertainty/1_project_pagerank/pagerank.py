import os
import random
import re
import sys
from copy import deepcopy
from functools import reduce

# In PageRank’s algorithm, a website is more important if it is linked to by other important websites,
# and links from less important websites have their links weighted less.

# Random Surfer Model:
# Surfer who starts with a web page at random, and then randomly chooses links to follow.
# The PageRank can be described as the probability that a random surfer is on that page at any given time.

# Damping Factor d:
# Probability the random surfer will choose from one of the links on the current page at random.
# Otherwise (with probability 1 - d), the random surfer chooses one out of all of the pages
# in the corpus at random (including the one they are currently on).
# Used to not get stuck at a certain range of the pages in the corpus.

# 1. Sampling Algorithm:
# The Model is interpreted as a Markov Chain, where each page represents a state,
# and each page has a transition model that chooses among its links at random.
# By using sampling and counting how many times a page was visited, the PageRank can be derived.
# 2. Iterative Algorithm:
# Use a formula that represents the Random Surfer Model to calculate the PageRank.
# The formula consists of 2 parts where the sum of both is PR(p).
# PR(p): PageRank of page p; d: damping factor; N: number of pages; NumLinks(i): number of links on page i
#   1. The surfer chooses one out of all of the pages: (1 - d) / N
#   2. The surfer followes a link from a page i to page p:
#   sum_over_pages_i( PR(i) / NumLinks(i) ) * d
#   That means when the surfer is on page i (the surfer is on page i with the probability of PR(i)),
#   there is a chance of 1 / NumLinks(i) that he will go to page p next.
# This formula can be solved iteratively. Initialize all PRs with 1 / N and iteratively update the PRs by
# continuously calculate new PRs with the formula.


# Damping Factor
DAMPING = 0.85
# Number of samples used for the Sampling Algorithm
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = samplePagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iteratePagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            # Ignore links from a page to itself
            pages[filename] = set(links) - {filename}

    # Remove links that don't point to a page in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def normalizeProbabilities(prob_distribution):
    """
    When calculating the PageRank by Sampling or Iteration, the resulting sum of all
    PageRank values might be slightly off from 1.
    Normalize all values, so they add up to 1.
    The input parameter is a dictionary where the values should add up to 1.
    """
    prob_sum = reduce(lambda x, y: x + y, list(prob_distribution.values()))
    for prob_key in prob_distribution:
        prob_distribution[prob_key] = prob_distribution[prob_key] / prob_sum


def transitionModel(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next, given a current page.
    Return a dictionary where keys are page names, and values are their visit probability.
    All probabilities should sum to 1.
    """
    transition_model = dict()
    linked_pages = corpus[page]
    if linked_pages:
        for page in corpus:
            # probability to choose a page out of all pages in the corpus
            transition_model[page] = (1 - damping_factor) / len(corpus)
            if page in linked_pages:
                # add probability to choose a page linked to by the current page
                transition_model[page] += damping_factor / len(linked_pages)
    # if the page has no links, choose between all pages in the corpus equally
    else:
        for page in corpus:
            transition_model[page] = 1 / len(corpus)

    return transition_model


def samplePagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = list(corpus)
    page_ranks = dict()
    for page in all_pages:
        page_ranks[page] = 0
    
    # choose random page to start
    page = random.choice(all_pages)
    page_ranks[page] += 1
    for _ in range(n - 1):
        # choose the next page according to the transition model
        transition_model = transitionModel(corpus, page, damping_factor)
        weights = []
        for page in all_pages:
            weights.append(transition_model[page])
        page = random.choices(all_pages, weights=weights)[0]
        page_ranks[page] += 1

    # convert the number of visits for each page to the proportion of all the samples
    # that corresponded to that page.
    for page in all_pages:
        page_ranks[page] = page_ranks[page] / n
    
    normalizeProbabilities(page_ranks)

    return page_ranks


def iteratePagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # If all PageRank values are changing no more than this threshold,
    # the iteration can be stopped.
    convergence_threshold = 0.001

    all_pages = list(corpus)
    # Interpret a page that has no links as having one link for every page in the corpus
    # (including itself). (As defined in the Random Surfer Transition Model).
    corpus_clean = deepcopy(corpus)
    for page in corpus_clean:
        if not corpus_clean[page]:
            corpus_clean[page] = set(all_pages)
    # initialize all PageRank values equally
    page_ranks = dict()
    for page in all_pages:
        page_ranks[page] = 1 / len(all_pages)
    while True:
        is_converged = True
        # update all PageRank values per Iteration
        for page in all_pages:
            page_rank_prev = page_ranks[page]
            # get all pages that have a link to the current page
            link_pages = [link_page for link_page in all_pages if page in corpus_clean[link_page]]
            # calculate probability that a surfer would follow a link to the current page
            link_to_page_prob = 0
            for link_page in link_pages:
                link_to_page_prob += page_ranks[link_page] / len(corpus_clean[link_page])
            # Random Surfer Model formula
            page_ranks[page] = (1 - damping_factor) / len(all_pages) + damping_factor * link_to_page_prob
            if abs(page_ranks[page] - page_rank_prev) > convergence_threshold:
                is_converged = False
        if is_converged:
            break
    
    normalizeProbabilities(page_ranks)

    return page_ranks


if __name__ == "__main__":
    main()
