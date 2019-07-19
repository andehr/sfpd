import heapq
from collections import namedtuple, OrderedDict
from enum import Enum

from sfpd.tokenise import PhraseTokeniser

import pandas as pd

class NgramCounter:
    """
    Data structure for counting occurrences of ngrams containing a root token in a tree structure similar to a word-trie.
    """

    def __init__(self, root_form,
                 min_n=1, max_n=6,
                 min_leaf_pruning=0.3,
                 min_ngram_count=4,
                 level1=5, level2=7, level3=15,
                 stopwords=None,
                 node_print_fn=None):
        """
        :param root_form: The root token around which the structure will be built.
        :param min_n: The minimum permitted ngram size
        :param max_n: The maximum permitted ngram size
        :param min_leaf_pruning: The threshold fraction of occurrence required for a longer ngram to not be filtered
        :param min_ngram_count: Minimum absolute count of an ngram
        :param level1:
        :param level2:
        :param level3:
        :param node_print_fn: Function applied to node in order to print it
        """
        self.level3 = level3
        self.level2 = level2
        self.level1 = level1
        self.min_ngram_count = min_ngram_count
        self.min_leaf_pruning = min_leaf_pruning
        self.max_n = max_n
        self.min_n = min_n
        self.root = Node(None, Arc.null(root_form), node_print_fn=node_print_fn, stopwords=stopwords)

    @property
    def root_form(self):
        return self.root.form

    def get_root_token_indices(self, context):
        return [idx for idx, token in enumerate(context) if token == self.root.form]

    def top_ngrams(self, k):
        """
        Get the top K ngrams given the current counts. WARNING: Involves pruning the tree.
        """
        self.root.prune_children(self.min_ngram_count, self.min_leaf_pruning, self.level1, self.level2, self.level3)

        ngrams = [(node.ngram(), node.count) for node in heapq.nlargest(k, self.root.get_nodes())]

        ngrams = [(ngram, count) for ngram, count in ngrams if len(ngram) >= self.min_n]

        if not ngrams and self.min_n <= 1:
            ngrams.append((self.root.form, self.root.count))

        return ngrams

    def add_context(self, context, weight=1):
        """
        Given a tokenised context, add the relevant counts to the data structure.

        :param context: Tokenised context containing examples of root token.
        :param weight: A context can have its contribution weighted.
        """
        root_idxs = self.get_root_token_indices(context)

        for root_idx in root_idxs:

            self.root.inc_count(weight)

            current_node = self.root
            last_before_node = self.root

            before_tokens = list(reversed(context[max(0, root_idx - self.max_n + 1):root_idx]))
            after_tokens = [] if len(context) - 1 == root_idx else context[root_idx + 1:min(root_idx + self.max_n,
                                                                                            len(context))]

            for after_token in after_tokens:
                current_node = current_node.inc_forward_child(after_token, weight)

            for idx, before_token in enumerate(before_tokens):
                current_node = last_before_node.inc_reverse_child(before_token, weight)
                last_before_node = current_node

                for idx2 in range(min(len(after_tokens), self.max_n - (idx + 2))):
                    current_node = current_node.inc_forward_child(after_tokens[idx2], weight)

# Useful as annotated node when analysing the datastructure in reverse from leafs up
Ancestor = namedtuple('Ancestor', ['node', 'childcount'])


class Node:
    """
    Represents a node in the trie-esque structure for counting ngrams. A node encapsulates a token which represents an
    ngram including that token and the root token.
    """

    def __init__(self, parent, to_parent, count=0, node_print_fn=None, stopwords=None):
        """
        :param parent: Parent node
        :param to_parent: Arc to parent node (forward/reverse/null)
        :param count: The count of the ngram represented by this node.
        :param node_print_fn: Function for printing this node in pretty output.
        """
        self.parent = parent
        self.to_parent = to_parent
        self.children = {}
        self.count = count
        self.depth = 0 if parent is None else parent.depth + 1
        self.node_print_fn = str if node_print_fn is None else node_print_fn
        self.stopwords = stopwords if stopwords else parent.stopwords

    def is_root(self):
        return self.parent is None or self.to_parent.is_null()

    def is_not_root(self):
        return not self.is_root()

    def get_stopword_count(self):
        return len([word for word in self.ngram() if word in self.stopwords])

    def endswith_stopword(self):
        return self.form in self.stopwords

    def __str__(self):
        return f"Node[{self.node_print_fn(self.form)}, p={self.parent}]"

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        # Normal order is that 'self' is shallower than 'other'
        if self.depth < other.depth:
            deeper = other
            shallower = self
            normal_order = True
        else:
            deeper = self
            shallower = other
            normal_order = False

        shallower_ancestors = shallower.ancestors_map()

        for ancestor in deeper.iter_ancestors():
            # if shallower is LCA, then favour deeper
            if ancestor.node == shallower:
                return True if normal_order else False

            if ancestor.node in shallower_ancestors:
                diff = shallower_ancestors[ancestor.node] - ancestor.childcount
                if diff == 0:
                    diff = deeper.get_stopword_count() - shallower.get_stopword_count()
                if diff == 0:
                    if shallower.endswith_stopword():
                        return True if normal_order else False
                    else:
                        return False if normal_order else True
                if diff < 0:
                    return True if normal_order else False
                else:
                    return False if normal_order else True

    def ancestors_map(self):
        if self.is_root():
            return {}

        ancestors = {}
        current_node = Ancestor(node=self.parent, childcount=self.count)
        while current_node.node.is_not_root():
            ancestors[current_node.node] = current_node.childcount
            current_node = Ancestor(node=current_node.node.parent, childcount=current_node.node.count)
        ancestors[current_node.node] = current_node.childcount  # add root
        return ancestors

    def iter_ancestors(self):
        current_node = self
        while current_node.is_not_root():
            yield Ancestor(node=current_node.parent, childcount=current_node.count)
            current_node = current_node.parent

    @property
    def form(self):
        return self.to_parent.form

    def has_parent(self):
        return self.parent is not None

    def has_children(self):
        return len(self.children) > 0

    def inc_count(self, increment):
        self.count = max(0, self.count + increment)

    def has_forward_child(self, form):
        return self.children[Arc.forward(form)]

    def has_reverse_child(self, form):
        return self.children[Arc.reverse(form)]

    def has_child(self, arc):
        return arc in self.children

    def add_forward_child(self, form, count):
        self.add_child(Arc.forward(form), count)

    def add_reverse_child(self, form, count):
        self.add_child(Arc.reverse(form), count)

    def add_child(self, arc, count):
        child = Node(self, arc, count, node_print_fn=self.node_print_fn)
        self.children[arc] = child
        return child

    def inc_forward_child(self, form, count):
        return self.inc_child(Arc.forward(form), count)

    def inc_reverse_child(self, form, count):
        return self.inc_child(Arc.reverse(form), count)

    def inc_child(self, arc, count):
        if self.has_child(arc):
            child = self.children[arc]
            child.inc_count(count)
            return child
        else:
            return self.add_child(arc, count)

    def prune_children(self, min_ngram_count, min_leaf_pruning, level1, level2, level3):
        # Remove all children with zero count
        self.children = {arc: child for arc, child in self.children.items() if child.count > 0}

        if self.children:
            # Process forward/reverse separately to make count proportions work
            forward = {arc: c for arc, c in self.children.items() if arc.is_forward()}
            reverse = {arc: c for arc, c in self.children.items() if arc.is_reverse()}

            forward = self.filter_children_by_count(forward, min_ngram_count, min_leaf_pruning, level1, level2, level3)
            reverse = self.filter_children_by_count(reverse, min_ngram_count, min_leaf_pruning, level1, level2, level3)

            self.children = {**forward, **reverse}

            for arc, child in self.children.items():
                child.prune_children(min_ngram_count, min_leaf_pruning, level1, level2, level3)

    def filter_children_by_count(self, children, min_ngram_count, min_leaf_pruning, level1, level2, level3):
        choices = len(children)
        total_occurrences = sum(child.count for child in children.values())

        if choices == total_occurrences:
            return {}

        dynamic_threshold = self.calc_dynamic_threshold(choices, total_occurrences, min_leaf_pruning, level1, level2,
                                                        level3)
        filtered = {}
        for arc, child in children.items():
            if child.count > min_ngram_count:
                proportion = child.count / self.count
                if proportion > 1 or proportion >= dynamic_threshold:
                    filtered[arc] = child
        return filtered

    def calc_dynamic_threshold(self, num_choices, total_occurrences, min_leaf_pruning, level1, level2, level3):
        if total_occurrences < level1:
            return 1.0
        elif total_occurrences < level2:
            return 0.75
        elif total_occurrences < level3:
            return 0.5
        else:
            return max(1.0 / num_choices, min_leaf_pruning)

    def ngram(self):
        if self.is_root():
            return [self.form]

        ancestors = []
        reverse_start = 0
        current_node = self
        while current_node.is_not_root():
            if current_node.to_parent.is_forward():
                reverse_start += 1
            ancestors.append(current_node.form)
            current_node = current_node.parent

        return ancestors[reverse_start:] + [current_node.form] + list(reversed(ancestors[0:reverse_start]))

    def print(self, prefix="", is_tail=True):
        print(
            prefix + self.arc_str(is_tail, self.to_parent.arc_type) + self.node_print_fn(self.form) + f"({self.count})")

        child_nodes = list(self.children.values())

        if child_nodes:
            for child in child_nodes[:-1]:
                child.print(prefix + ("    " if is_tail else "│   "), False)

            child_nodes[-1].print(prefix + ("    " if is_tail else "│   "), True)

    def arc_str(self, is_tail, arc_type):
        if arc_type is ArcType.FORWARD:
            return "└>─ " if is_tail else "├>─ "
        elif arc_type is ArcType.REVERSE:
            return "└<─ " if is_tail else "├<─ "
        else:
            return "└── " if is_tail else "├── "

    def get_nodes(self):
        found = [self]
        explore = [self]
        while explore:
            current_node = explore.pop()
            for child in current_node.children.values():
                if child.has_children():
                    explore.append(child)
                found.append(child)
        return found


ArcType = Enum("ArcType", ["FORWARD", "REVERSE", "NULL"])


class Arc:

    def __init__(self, form, arc_type):
        self.form = form
        self.arc_type = arc_type

    def is_forward(self):
        return self.arc_type is ArcType.FORWARD

    def is_reverse(self):
        return self.arc_type is ArcType.REVERSE

    def is_null(self):
        return self.arc_type is ArcType.NULL

    def __str__(self):
        return f"Arc[{self.arc_type}, {self.form}]"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Arc):
            return self.form == other.form and self.arc_type == other.arc_type
        return False

    def __hash__(self):
        return hash((self.form, self.arc_type))

    @staticmethod
    def forward(form):
        return Arc(form, ArcType.FORWARD)

    @staticmethod
    def reverse(form):
        return Arc(form, ArcType.REVERSE)

    @staticmethod
    def null(form):
        return Arc(form, ArcType.NULL)


def get_top_phrases(words, texts, k=1, language="en",
                    min_n=1, max_n=6,
                    min_leaf_pruning=0.3,
                    min_ngram_count=4,
                    level1=5, level2=7, level3=15):

    tokeniser = PhraseTokeniser(language)
    stopwords = tokeniser.get_stopwords()

    print("> Finding phrases")

    counters = [NgramCounter(word,
                             min_n, max_n,
                             min_leaf_pruning,
                             min_ngram_count,
                             level1, level2, level3,
                             stopwords) for word in words]

    doc_count = 0
    for text in texts:
        tokens = tokeniser(text)

        for counter in counters:
            counter.add_context(tokens, 1)

        doc_count += 1
        if doc_count % 100 == 0:
            print(f"\r> Processed {doc_count} docs", end="")
    print()

    return TopPhrases(counters, k)


class TopPhrases:

    def __init__(self, counters, num_phrases):
        """
        Each word is associated with 1 or more phrases. The ordering of most surprising word first is maintained.
        """
        self.data = OrderedDict((c.root_form, c.top_ngrams(num_phrases)) for c in counters)

    def raw_phrases(self):
        """
        For each surprising word in turn, collect all of its phrases, all together in one big list.
        """
        return [self.phrase2str(phrase[0]) for word, phrases in self.data.items() for phrase in phrases]

    def top_phrases_per_word(self):
        return OrderedDict((word, phrases[0]) for word, phrases in self.data.items())

    def phrase2str(self, phrase):
        return " ".join(phrase)


