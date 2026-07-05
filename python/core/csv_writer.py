import csv


class CSVWriter:

    @staticmethod
    def save(filename, x, y):

        with open(filename, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["Input", "Output"])

            for a, b in zip(x, y):

                writer.writerow([a, b])