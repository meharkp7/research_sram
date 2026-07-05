import matplotlib.pyplot as plt


class Plotter:

    @staticmethod
    def plot(x, y, xlabel, ylabel, title):

        plt.figure(figsize=(10,5))

        plt.plot(time, vin, label="VIN")
        plt.plot(time, vout, label="VOUT")

        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Transient Response")

        plt.grid(True)
        plt.legend()

        plt.savefig("../results/vtc.png", dpi=300)
        
        plt.show()