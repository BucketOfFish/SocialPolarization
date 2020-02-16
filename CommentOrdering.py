import numpy as np
import matplotlib.pyplot as plt


class Individual:
    def __init__(self):
        self.opinion = np.random.normal(0, 0.2)
        self.politeness = np.random.normal(0, 0.2)
        self.openness = np.random.normal(0.5, 0.1)
        self.read_max_comments_in_level = np.random.randint(1, 21)

    def read_and_reply(self, head_comment):
        # read comments and vote - comments will shift order during voting, so prefetch the comments to read
        read_list = []
        comment = head_comment.next
        n_read_comments = 0
        while comment is not None and n_read_comments < self.read_max_comments_in_level:
            read_list.append(comment)
            n_read_comments += 1
            comment = comment.next
        for comment in read_list:
            self.vote(comment)
            self.change_opinion(comment)
            # decide whether to reply, move on to the next comment, or read replies
            if self.decide_to_reply(comment):
                comment.add_reply(self.opinion, self.politeness)
                return
            elif self.decide_to_read_replies(comment):
                self.read_and_reply(comment.child_head)
                return
        # leave a comment at the last level we reached if we haven't already
        head_comment.add_reply(self.opinion, self.politeness)

    def vote(self, comment):
        match_opinion = comment.opinion * self.opinion > 0
        p_upvote = match_opinion
        p_downvote = not match_opinion
        if np.random.uniform() < p_upvote:
            comment.vote(1)
        if p_upvote < np.random.uniform() < p_downvote:
            comment.vote(-1)
        else:
            comment.vote(0)

    def change_opinion(self, comment):
        self.opinion = min(max(comment.opinion*self.openness*comment.politeness + self.opinion, -1), 1)

    def decide_to_reply(self, comment):
        opinion_diff = abs(comment.opinion - self.opinion)
        p_reply = (opinion_diff > 1) or (opinion_diff < 0.2)
        return np.random.uniform() < p_reply

    def decide_to_read_replies(self, comment):
        if comment.has_replies():
            p_read = np.exp(-comment.get_n_total_replies()/20)
            return np.random.uniform() < p_read
        else:
            return False


class Population:
    def __init__(self, n_individuals):
        self.population = []
        for i in range(n_individuals):
            self.population.append(Individual())

    def plot_opinions(self, name):
        opinions = [i.opinion for i in self.population]
        plt.hist(opinions, bins=100)
        plt.xlim(-1, 1)
        plt.savefig(name)
        plt.clf()

    def plot_politeness(self, name):
        politeness = [i.politeness for i in self.population]
        plt.hist(politeness, bins=100)
        plt.savefig(name)
        plt.clf()

    def plot_bias(self, name):
        bias = [i.openness for i in self.population]
        plt.hist(bias, bins=100)
        plt.savefig(name)
        plt.clf()

    def __iter__(self):
        for individual in self.population:
            yield individual


class Comment:
    def __init__(self, opinion, politeness, *, head, previous=None, next=None):
        self.opinion = opinion
        self.politeness = politeness
        self.previous = previous
        self.next = next
        self.head = head
        self.parent = head.parent
        self.child_head = None
        self.depth = head.depth+1
        self.karma = 0

    def is_head(self):
        return False

    def get_n_total_replies(self):
        if self.child_head is None:
            return 0
        else:
            return self.child_head.n_total_replies

    def has_replies(self):
        return self.child_head is not None

    def vote(self, vote):
        self.karma += vote
        if not self.previous.is_head() and self.previous.karma < self.karma:
            parent = self.previous
            grandparent = parent.previous
            child = self.next
            parent.next = child
            if child is not None:
                child.previous = parent
            self.next = parent
            self.previous = grandparent
            grandparent.next = self
        if self.next is not None and self.next.karma > self.karma:
            parent = self.previous
            child = self.next
            grandchild = child.next
            parent.next = child
            child.previous = parent
            self.next = grandchild
            self.previous = child
            child.next = self

    def add_reply(self, opinion, politeness):
        if self.child_head is None:
            self.child_head = HeadComment(parent=self, thread=self.head.thread, depth=self.head.depth+1)
        self.child_head.add_reply(opinion, politeness)

    def __str__(self):
        spaces = " " * self.depth
        print(f"{spaces}({self.karma}, {self.openness})")
        if self.child_head is not None:
            print(self.child_head)
        if self.next is not None:
            print(self.next)


class HeadComment:
    '''This class is a dummy comment, treated as the first 'comment' at any given depth in the thread.
       It contains information about the comments at that depth.'''
    def __init__(self, *, parent, thread, depth):
        self.next = None
        self.parent = parent
        self.n_direct_replies = 0
        self.n_total_replies = 0
        self.depth = depth
        self.thread = thread

    def is_head(self):
        return True

    def get_n_direct_replies(self):
        return self.n_direct_replies

    def get_n_total_replies(self):
        return self.n_total_replies

    def add_reply(self, opinion, politeness):
        self.n_direct_replies += 1
        self.increment_total_replies()

        comment_pointer = self
        while comment_pointer.next is not None and comment_pointer.next.karma > 0:
            comment_pointer = comment_pointer.next
        new_comment = Comment(opinion, politeness, head=self, previous=comment_pointer, next=comment_pointer.next)
        comment_pointer.next = new_comment
        if new_comment.next is not None:
            new_comment.next.previous = new_comment

    def increment_total_replies(self):
        self.n_total_replies += 1
        if self.parent is not None:
            self.parent.head.increment_total_replies()

    def has_replies(self):
        return self.n_direct_replies == 0

    def __str__(self):
        self.next.display()


class Thread:
    def __init__(self):
        '''Initialize the thread with a 0-depth head comment with no replies.'''
        self.head_comment = HeadComment(parent=None, thread=self, depth=0)

    def fill_comments(self, population):
        '''Get everybody in the population to read the thread and leave a comment.'''
        for individual in population:
            individual.read_and_reply(self.head_comment)

    def __str__(self):
        if self.head_comment.has_replies():
            print(self.head_comment)
        else:
            print("No comments in thread")


if __name__ == "__main__":
    n_individuals = 10000
    n_threads = 50

    population = Population(n_individuals)

    population.plot_politeness("Plots/initial_politeness.eps")
    population.plot_bias("Plots/initial_bias.eps")
    population.plot_opinions("Plots/initial_opinions.eps")

    for i in range(n_threads):
        thread = Thread()
        thread.fill_comments(population)
        population.plot_opinions(f"Plots/opinions_{i}.eps")