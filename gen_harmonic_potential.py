# ！/usr/local/python3.7
# learning the OOP method
import numpy as np
import matplotlib as plt


class Particle:
    def __init__(self, ntype, nl):
        self.table = np.zeros((nl ** 2, 2), int)
        self.ntype = ntype
        self.num = nl ** 2
        for i in range(nl ** 2):
            self.table[i, 0] = i + 1
            self.table[i, 1] = (i % ntype) + 1
        #           print("id: %d type: %d" % (self.id, self.type))
        # print(self.table)

    def insert_ghost(self):
        self.table = np.row_stack((self.table, [self.num + 1, self.ntype + 1]))


p = Particle(128, 32)
p.insert_ghost()
print(p.table)
