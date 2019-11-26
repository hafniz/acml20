import os, numpy
from scipy import stats

filenames = []
for root, dirs, files in os.walk("..\\Dataset\\artificial-new\\level0\\rebalanced\\dataset"):
    for file in files:
        filenames.append(os.path.join(root, file))

# -inf | 0.135% | m-3s | 2.14% | m-2s | 13.59% | m-s | 34.135% | m | 34.135% | m+s | 13.59% | m+2s | 2.14% | m+3s | 0.135% | +inf
expectedFreq = [0.00135, 0.0214, 0.1359, 0.34135, 0.34135, 0.1359, 0.0214, 0.00135]
fields = ["filename,chi-sq,p-value,skewness,kurtosis"]

for filename in filenames:
    alphas = []
    with open(filename) as file:
        alphas = [float(row.split(',')[-1]) for row in list(file)[1:]]

    count = len(alphas)
    mean = numpy.mean(alphas)
    stddev = numpy.std(alphas)

    observedFreq = [0 for _ in range(len(expectedFreq))]
    for alpha in alphas:
        if alpha < mean - 3 * stddev:
            observedFreq[0] += 1
        elif alpha < mean - 2 * stddev:
            observedFreq[1] += 1
        elif alpha < mean - stddev:
            observedFreq[2] += 1
        elif alpha < mean:
            observedFreq[3] += 1
        elif alpha < mean + stddev:
            observedFreq[4] += 1
        elif alpha < mean + 2 * stddev:
            observedFreq[5] += 1
        elif alpha < mean + 3 * stddev:
            observedFreq[6] += 1
        else:
            observedFreq[7] += 1
    observedFreq = [catFreq / count for catFreq in observedFreq]
    chiSquared = sum([(observedFreq[i] - expectedFreq[i]) ** 2 / expectedFreq[i] for i in range(len(expectedFreq))])
    pValue = 1 - stats.chi2.cdf(chiSquared, len(expectedFreq) - 1)
    
    skewness = sum([(alpha - mean) ** 3 for alpha in alphas]) / count / (stddev ** 3)
    kurtosis = sum([(alpha - mean) ** 4 for alpha in alphas]) / count / (stddev ** 4) - 3

    fields.append(f"{os.path.basename(filename).split('.')[0]},{chiSquared},{pValue},{skewness},{kurtosis}")
    print(f"Successfully finished {filename}")

with open("normality_skewness_and_kurtosis.csv", 'w') as file:
    file.write('\n'.join(fields))
