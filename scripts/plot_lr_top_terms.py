import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../outputs/models/tfidf_lr/top_terms.csv")

# Top REAL terms
plt.figure(figsize=(10, 6))
plt.barh(df["top_real_terms"][:15], df["real_coef"][:15], color="green")
plt.title("Top 15 Terms Indicating REAL News — Logistic Regression")
plt.xlabel("Coefficient Weight")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../outputs/lr_top_terms_real.png", dpi=300)
plt.show()

# Top FAKE terms
plt.figure(figsize=(10, 6))
plt.barh(df["top_fake_terms"][:15], df["fake_coef"][:15], color="red")
plt.title("Top 15 Terms Indicating FAKE News — Logistic Regression")
plt.xlabel("Coefficient Weight")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../outputs/lr_top_terms_fake.png", dpi=300)
plt.show()

print("Saved: lr_top_terms_real.png & lr_top_terms_fake.png")
