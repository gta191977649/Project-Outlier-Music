import matplotlib.pyplot as plt


a = [1,5,6,4]
plt.figure(figsize=(3,1))
plt.plot(a,ds='steps-mid',color="b")
plt.tight_layout()
plt.show()