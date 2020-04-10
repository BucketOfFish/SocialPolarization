import numpy as np
import matplotlib.pyplot as plt
import random


class Individual:
    def __init__(self, is_moderator):
        if (np.random.uniform() < 0.1 or is_moderator):
            self.opinion = np.clip(np.random.normal(-0.7, 0.1), -1, 1)
        else:
            self.opinion = np.clip(np.random.normal(0, 0.1), -1, 1)
        self.bias = np.clip(np.random.normal(0.8, 0.05), 0, 1)
        self.openness = np.clip(np.random.normal(0.3, 0.05), 0, 1)
        self.patience = np.random.poisson(5)
        self.persuasiveness = np.clip(np.random.normal(0.4, 0.1), 0, 1)

    def read_and_reply(self, head_comment):
        # read comments and vote - comments will shift order during voting, so prefetch the comments to read
        read_list = head_comment.get_top_replies(self.patience, self.opinion)
        for comment in read_list:
            self.vote(comment)
            self.change_opinion(comment)
            # decide whether to reply, move on to the next comment, or read replies
            if self.decide_to_reply(comment):
                comment.add_reply(self.opinion, self.get_comment_persuasiveness())
                return
            elif self.decide_to_read_replies(comment):
                self.read_and_reply(comment)
                return
        # leave a comment at the last level we reached if we haven't already
        head_comment.add_reply(self.opinion, self.get_comment_persuasiveness())

    def get_comment_persuasiveness(self):
        def get_gaussian_persuasiveness():
            return np.clip(np.random.normal(0.2, 0.1), 0, 1)

        def get_truth_biased_persuasiveness():
            truth_value = 0.7
            truth_boost = 0.5 - abs(truth_value - self.opinion)
            persuasiveness = get_gaussian_persuasiveness() + truth_boost
            return np.clip(persuasiveness, 0, 1)

        def get_self_persuasiveness():
            return self.persuasiveness

        return get_self_persuasiveness()

    def vote(self, comment):
        opinion_diff = abs(comment.opinion - self.opinion)
        if (opinion_diff < 0.6) and (opinion_diff > 0.2):
            comment.vote(0)
        else:
            p_upvote = np.exp(-opinion_diff * 5)
            if np.random.uniform() < p_upvote:
                comment.vote(1)
            else:
                comment.vote(-1)

    def change_opinion(self, comment):
        def get_team_opinion():
            opinion_diff = abs(comment.opinion - self.opinion)
            match_opinion = comment.opinion * self.opinion > 0
            if match_opinion:
                persuasiveness = comment.persuasiveness + self.bias * opinion_diff
            else:
                persuasiveness = comment.persuasiveness - self.bias * opinion_diff
            return min(max(comment.opinion*self.openness*persuasiveness + self.opinion, -1), 1)

        def get_biased_opinion():
            opinion_diff = comment.opinion - self.opinion
            persuasiveness = comment.persuasiveness - self.bias * abs(opinion_diff)
            return min(max(opinion_diff*self.openness*persuasiveness + self.opinion, -1), 1)

        self.opinion = get_biased_opinion()

    def decide_to_reply(self, comment):
        opinion_diff = abs(comment.opinion - self.opinion)
        if (opinion_diff > 1):
            p_reply = opinion_diff/2
        elif (opinion_diff < 0.2):
            p_reply = (0.2 - opinion_diff) * 5
        else:
            p_reply = 0
        return np.random.uniform() < p_reply

    def decide_to_read_replies(self, comment):
        if comment.has_replies():
            p_read = 0.2 + 0.8 * (1 - np.exp(-comment.n_total_replies/20))
            return np.random.uniform() < p_read
        else:
            return False


class Population:
    def __init__(self, n_individuals):
        self.population = []
        for i in range(n_individuals):
            self.population.append(Individual(i == 0))

    def plot_opinions(self, name):
        opinions = [i.opinion for i in self.population]
        plt.hist(opinions, bins=100)
        plt.xlim(-1, 1)
        plt.savefig(name)
        plt.clf()

    def plot_openness(self, name):
        openness = [i.openness for i in self.population]
        plt.hist(openness, bins=100)
        plt.xlim(0, 1)
        plt.savefig(name)
        plt.clf()

    def plot_bias(self, name):
        bias = [i.bias for i in self.population]
        plt.hist(bias, bins=100)
        plt.xlim(0, 1)
        plt.savefig(name)
        plt.clf()

    def __iter__(self):
        for individual in self.population:
            yield individual


class Comment:
    def __init__(self, opinion, persuasiveness, *, depth, parent=None, thread):
        self.opinion = opinion
        self.persuasiveness = persuasiveness
        self.parent = parent
        self.children = []
        self.thread = thread
        self.depth = depth
        self.karma = 0
        self.n_direct_replies = 0
        self.n_total_replies = 0

    def has_replies(self):
        return self.n_total_replies > 0

    def vote(self, vote):
        self.karma += vote

    def add_reply(self, opinion, persuasiveness):
        def no_moderation():
            return False

        def objective_moderation():
            return persuasiveness > 0.8

        def biased_moderation():
            # the first person in the population is the moderator
            opinion_diff = opinion - self.thread.moderator.opinion
            subjective_persuasiveness = persuasiveness - self.thread.moderator.bias * abs(opinion_diff)
            return subjective_persuasiveness > 0.8

        gets_moderated = no_moderation()
        if (gets_moderated):
            return
        else:
            new_comment = Comment(opinion, persuasiveness, depth=self.depth+1, parent=self, thread=self.thread)
            self.children.append(new_comment)
            self.n_direct_replies += 1
            self.increment_total_replies()

    def increment_total_replies(self):
        self.n_total_replies += 1
        if self.parent is not None:
            self.parent.increment_total_replies()

    def get_top_replies(self, n_replies, opinion):
        def get_top_voted_replies():
            if self.n_direct_replies <= n_replies:
                return self.children
            else:
                read_list = []
                karma = np.array([child.karma for child in self.children])
                top_voted_indices = np.argpartition(karma, -n_replies)[-n_replies:]
                for index in top_voted_indices:
                    read_list.append(self.children[index])
                return read_list

        def get_random_replies():
            if self.n_direct_replies <= n_replies:
                return self.children
            else:
                read_indices = random.sample(range(self.n_direct_replies), n_replies)
                read_list = [self.children[i] for i in read_indices]
                return read_list

        def get_similar_replies():
            if self.n_direct_replies <= n_replies:
                return self.children
            else:
                similarities = [abs(child.opinion - opinion) for child in self.children]
                ss = [abs(i - 0.1) for i in similarities]
                exp_ss = [np.exp(-5*i) for i in ss]
                exp_ss /= sum(exp_ss)

                read_indices = np.random.choice(range(self.n_direct_replies), n_replies, p=exp_ss)
                read_list = [self.children[i] for i in read_indices]
                return read_list

        return get_similar_replies()

    def display(self):
        spaces = " " * self.depth
        print(f"{spaces}({self.karma}, {self.opinion:.2f})")
        for child in self.children:
            child.display()


class Thread:
    def __init__(self):
        '''Initialize the thread with a 0-depth head comment with no replies.'''
        self.head_comment = Comment(0, 0, depth=0, parent=None, thread=self)

    def fill_comments(self, population):
        '''Get everybody in the population to read the thread and leave a comment.'''
        for individual in population:
            individual.read_and_reply(self.head_comment)

    def display(self):
        if self.head_comment.has_replies():
            self.head_comment.display()
        else:
            print("No comments in thread")


if __name__ == "__main__":
    n_individuals = 10000
    n_threads = 25

    population = Population(n_individuals)

    # population.plot_openness("Plots/openness.eps")
    # population.plot_bias("Plots/bias.eps")

    for i in range(n_threads):
        print(f'Populating comment thread {i}/{n_threads}', end='\r')
        thread = Thread()
        thread.moderator = population.population[0]
        if (i % 5 == 0):
            population.plot_opinions(f"Plots/opinions_{i}.eps")
        thread.fill_comments(population)
    population.plot_opinions(f"Plots/opinions_{n_threads}.eps")
