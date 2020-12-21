import os

print(os.getcwd())

file1 = open('/home/michael/Desktop/git/ceds-cigar/pycigar/oscillation/IEEE_37_feeder_UB/IEEE_37_Bus_allwye_noxfm_noreg.dss', 'r')
file2 = open('/home/michael/Desktop/git/ceds-cigar/pycigar/oscillation/IEEE_37_feeder_UB/IEEE_37_Bus_allwye_noxfm_noreg_temp.dss', 'w')

for aline in file1:

    if 'New Line.' in aline:
        print(aline)
        arr1 = aline.split(' ')
        for k1 in range(len(arr1)):

            arr2 = arr1[k1].split('.')
            if arr2[0] == 'Line':
                print(arr2[1])

            arr2 = arr1[k1].split('=')
            if arr2[0] == 'Bus1':
                print(arr2[1].split('.')[0])
                bus1 = arr2[1].split('.')[0]

            arr2 = arr1[k1].split('=')
            if arr2[0] == 'Bus2':
                print(arr2[1].split('.')[0])
                bus2 = arr2[1].split('.')[0]

        newname = 'line_' + bus1 + '_' + bus2
        print(newname)

        x = aline.split('Line.')
        print(x[0])
        print(x[1])
        newline = 'New Line.' + newname + ' ' + x[1].split(' ',1)[1]
        print(newline)
        file2.write(newline)
    else:
        file2.write(aline)


file1.close()
file2.close()