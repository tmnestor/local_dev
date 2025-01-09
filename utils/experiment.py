#!/usr/bin/env python3
"""
Utility module for running experiments.
"""

class Experiment:
    def __init__(self, name: str):
        self.name = name
        self.results = {}

    def run(self):
        """Run the experiment"""
        pass

    def add_result(self, key: str, value: any):
        """Add a result to the experiment"""
        self.results[key] = value

    def get_results(self):
        """Get all experiment results"""
        return self.results

def main():
    # Example usage
    exp = Experiment("test_experiment")
    exp.run()
    exp.add_result("success", True)
    print(exp.get_results())

if __name__ == "__main__":
    main()