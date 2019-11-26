stats = ["midAlphaLt,count,R11,R12,R13,R14,KNN,NB,DT-M,DT-R,LS,RT"]

rows = []
with open("no_mid_alphas.csv") as file:
    rows = [row[:-1].split(',') for row in list(file)[1:]]

for threshold in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]:
    count = R11 = R12 = R13 = R14 = KNN = NB = DT_M = DT_R = LS = RT = 0
    for row in rows:
        if float(row[3]) >= threshold:
            continue
        count += 1
        if row[0] == "R11":
            R11 += 1
        if row[0] == "R12":
            R12 += 1
        if row[0] == "R13":
            R13 += 1
        if row[0] == "R14":
            R14 += 1
        if row[2] == "KNN":
            KNN += 1
        if row[2] == "NB":
            NB += 1
        if row[2] == "DT-M":
            DT_M += 1
        if row[2] == "DT-R":
            DT_R += 1
        if row[1].startswith("LS"):
            LS += 1
        if row[1].startswith("RT"):
            RT += 1
    stats.append(f"{threshold},{count},{R11},{R12},{R13},{R14},{KNN},{NB},{DT_M},{DT_R},{LS},{RT}")

with open("mid_alpha_stats.csv", 'w') as file:
    file.write('\n'.join(stats))
