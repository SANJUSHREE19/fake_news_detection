import pandas as pd
import matplotlib.pyplot as plt

real = pd.read_csv("../data/FakeReal/True.csv")
fake = pd.read_csv("../data/FakeReal/Fake.csv")

counts = {"Real": len(real), "Fake": len(fake)}

plt.bar(counts.keys(), counts.values(), color=["green", "red"])
plt.title("Dataset Class Distribution (Fake vs Real News)")
plt.ylabel("Number of Samples")
plt.savefig("../outputs/class_distribution.png", dpi=300)
plt.show()
print("Saved to outputs/class_distribution.png")
