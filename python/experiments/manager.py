"""
Experiment Manager
"""


class ExperimentManager:

    def __init__(self):

        self.experiments = []

    def add(self, experiment):

        self.experiments.append(experiment)

    def run_all(self):

        all_results = []

        for experiment in self.experiments:

            print("=" * 60)
            print(type(experiment).__name__)
            print("=" * 60)

            experiment.prepare()

            result = experiment.run()

            experiment.analyze()

            all_results.append(result)

        return all_results