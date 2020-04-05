"""Print out information about sampled architectures from the log file"""

import sys
from collections import Counter
import operator

filename = sys.argv[1]
uqs = []
n_total = 0
k = 10

with open(filename, "rb") as f:
    for l in f.readlines():
        arch = l.decode("utf-8").strip("\n").split(":")[-1]
        reward = float(l.decode("utf-8").strip("\n").split(",")[0][7:])
        epoch = int(l.decode("utf-8").strip("\n").split(":")[2].split(",")[0])
        # uqs.add(arch)
        uqs.append((arch, (reward, epoch)))
        n_total += 1
c_uqs = Counter(elem[0] for elem in uqs)
top_k = c_uqs.most_common(k)
print("Unique {} out of {}".format(len(c_uqs), n_total))
print("-" * 15)
print("Top-{} most common".format(k))
for e in top_k:
    print(e)
print("-" * 15)
print("Top-{} with highest reward".format(k))
uqs.sort(key=operator.itemgetter(1), reverse=True)
for e in uqs[:k]:
    print(e)
