from scipy import stats 

def WriteTTestResults(accuracyFilename, outputFilename):
    lines = []
    output = [ "MSetName,betterModel,t-value,p-value" ]
    with open(accuracyFilename) as file:
        lines = list(file)
    for i in range(1, len(lines)):
        fields = lines[i].split(',')
        tValue, pValue = stats.ttest_ind([float(s) for s in fields[1:11]], [float(s) for s in fields[11:21]])

        if pValue > 0.05:
            # cannot reject the null hypothesis of identical average scores => draw
            output.append(f"{fields[0]},Draw,{tValue},{pValue}")
        else:
            # reject the null hypothesis of equal averages => one is better than another
            betterModel = "Beta" if tValue > 0 else "Conventional"
            output.append(f"{fields[0]},{betterModel},{tValue},{pValue}")
    with open(outputFilename, 'w') as file:
        file.write('\n'.join(output))
