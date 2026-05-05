import numpy

for epoch in range(0,100):
    alpha = 2. / (1. + numpy.exp(-0.1* (epoch - 100 / 3)))
    print(f"{alpha}\t{0.01*epoch}\t{0.05*epoch}")