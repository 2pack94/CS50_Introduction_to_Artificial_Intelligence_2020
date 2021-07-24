import csv
import itertools
import sys
from functools import reduce

# Heredity of the GJB2 gene (causes hearing impairment)

# Each person possesses 2 copies of the gene where each gene is either active or inactive.
# Each person can possess either 0, 1, or 2 copies of the active gene (called "gene" from now on).
# Depending on the number of genes there is a different probability to have the trait.
# Every child inherits one copy of the gene from each of their parents.
# Inherit probability:
#   0 if no parent has it
#   0.5 if only 1 parent has it
#   1 if both parents have it

# The relationships can be modeled as a Bayesian Network.
# Every Person has 2 Nodes (random variables): PersonGene (0, 1, 2) and PersonTrait (True, False)
# PersonGene is dependent on PersonGene of both parents. If the parents are not known,
# PersonGene will be determined by the unconditional probability to have the gene.
# PersonTrait is only dependent on PersonGene of the same person.

# Mutation Probability:
# When a gene is inherited it has a probability of 0.01 to mutate from an active version
# to an inactive version and vice versa.
# The mutation probability only needs be considered when a parent has 2 or no copies of the gene.
# If a parent has 1 copy of the gene, the mutation probabilities from inactive -> active
# and active -> inactive are cancelling each other out.
# Inheritance probability 1 gene: 0.5 * 0.01 + 0.5 * 0.99 = 0.5

# The csv files store the information name, mother, father, trait but not all information are known.
# The information determines the hierarchy of the Network and which person
# has a known trait (evidence).

# The goal is to calculate the probability distribution of all Nodes in the Network.
# The method used is inference by enumeration (see 0_lecture/0_bayesnet/1_inference.py).
# All hidden variables are enumerated.
# For each enumeration a joint probability is calculated (see 0_lecture/0_bayesnet/0_likelihood.py).
# All joint probabilities are added. The probability of each value of each Node is filled
# to get its probability distribution.
# Each probability distribution is normalized to add up to 1.


PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = loadData(sys.argv[1])

    # Keep track of gene and trait probabilities for each person.
    # create with dictionary comprehension
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information.
        # If the trait is known then the person must be correctly present ot absent
        # in the have_trait list, depending on if trait is True of False.
        # A known trait can be used as evidence.
        # All other variables are hidden and need to be enumerated.
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                joint_p = jointProbability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, joint_p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def loadData(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    When mother and father are known, they must have their own entry.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def jointProbability(people, one_gene, two_genes, have_trait):
    """
    Return a joint probability (function is called for each enumeration).
    The value of each individual probability (probability that the given value of the variable is true)
    is multiplied to get the joint probability.
    The probability returned is the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set `have_trait` does not have the trait.
    """
    probs = []
    no_gene = set(people) - one_gene - two_genes
    for person in people:
        num_genes = 0
        is_trait = False
        gene_prob = 0
        if person in no_gene:
            num_genes = 0
        elif person in one_gene:
            num_genes = 1
        elif person in two_genes:
            num_genes = 2
        if person in have_trait:
            is_trait = True
        else:
            is_trait = False
        
        parents = [people[person]["mother"], people[person]["father"]]
        # If parents are known calculate conditional probability of having
        # the given number of genes.
        if parents[0] and parents[1]:
            parent_heredity = {
                parents[0]: 0,
                parents[1]: 0
            }
            for parent in parents:
                if parent in no_gene:
                    parent_heredity[parent] = 0 + PROBS["mutation"]
                elif parent in one_gene:
                    parent_heredity[parent] = 0.5
                elif parent in two_genes:
                    parent_heredity[parent] = 1 - PROBS["mutation"]
            if num_genes == 0:
                # A child does not have the gene if it doesn't get it from both parents.
                gene_prob = (1 - parent_heredity[parents[0]]) * (1 - parent_heredity[parents[1]])
            elif num_genes == 1:
                # A child does have one gene if it does get it from one parent but not from the other.
                # (mother gene and not father gene) or (not mother gene and father gene)
                gene_prob = \
                    parent_heredity[parents[0]] * (1 - parent_heredity[parents[1]]) + \
                    (1 - parent_heredity[parents[0]]) * parent_heredity[parents[1]]
            elif num_genes == 2:
                # A child does have two genes if it does get it from both parents.
                gene_prob = parent_heredity[parents[0]] * parent_heredity[parents[1]]
        # Use unconditional probability if the parents are not known
        else:
            gene_prob = PROBS["gene"][num_genes]

        # The trait probability depends on the persons genes.
        trait_prob = PROBS["trait"][num_genes][is_trait]

        probs.append(gene_prob)
        probs.append(trait_prob)

    # multiply all individual probabilities to get the joint probability
    joint_prob = reduce(lambda x, y: x * y, probs)
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, joint_prob):
    """
    Add to `probabilities` a new joint probability `joint_prob` (function is called for each enumeration).
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on in which gene list
    the person is and if the person is in `have_trait`, respectively
    """
    people = set(probabilities)
    no_gene = people - one_gene - two_genes
    for person in people:
        num_genes = 0
        is_trait = False
        if person in no_gene:
            num_genes = 0
        elif person in one_gene:
            num_genes = 1
        elif person in two_genes:
            num_genes = 2
        if person in have_trait:
            is_trait = True
        else:
            is_trait = False

        probabilities[person]["gene"][num_genes] += joint_prob
        probabilities[person]["trait"][is_trait] += joint_prob


def normalize(probabilities):
    """
    Update `probabilities` such that the probability distribution for "gene" and "trait"
    for each person is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        for distribution in ["gene", "trait"]:
            prob_sum = reduce(lambda x, y: x + y, list(probabilities[person][distribution].values()))
            for prob_key in probabilities[person][distribution]:
                probabilities[person][distribution][prob_key] = \
                    probabilities[person][distribution][prob_key] / prob_sum


if __name__ == "__main__":
    main()
