import sys
####### Extract Data ###########################################################################################
def returnMETADATA(filename):
    N = 0
    N_c = []
    n_c = []

    META = {}
    META_main = {}

    i = -1
    last_family = 'name'
    with open(filename) as fp:
        for line in fp:
            N += 1
            data_id = N
            i += 1
            try:
                (sep_l, sep_w, pet_l, pet_w, family) = line.split(',')
            except:
                print('FILE-ERROR: Ensure no spaces are left at the end of the data file')
                sys.exit(0)
                
            family = family.rstrip('\n') # remove \n from new line
            
            if last_family != family: # 
                N_c.append(family)
                n_c.append(i)
                i=0
            META_main[data_id] = {'sepalLength':sep_l, 'sepalWidth':sep_w, 'petalLength':pet_l, 'petalWidth':pet_w}
            last_family = family
    n_c.append(N-sum(n_c))
    n_c.remove(0)
    for i in range(len(n_c)):
        META[N_c[i]] = {}
        for j in range(n_c[i]):
            key = ((i)*n_c[i]+j)+1
            temp = META_main[key]
            new_id = N_c[i] + str(j)
            new_data = {j:temp}
            META[N_c[i]].update(new_data)
    count = 0
    return META
