allAttent = np.load('TransFormer/allAttent.npy')[5]   #get the layer

tempatten = []

for dim0 in range(500):
    tempatten.append(torch.from_numpy(np.cumsum(allAttent[dim0], axis=0)[-1]).argmax(dim = 1).tolist())
print(len(tempatten[0]))

#get the argmax

de = open('D:\My_Days\My_code\dataset\DeEn\\de','rb')
en = open('D:\My_Days\My_code\dataset\DeEn\\en','rb')

output = open('D:\My_Days\My_code\dataset\DeEn\\layer6attentionalign', 'a')

index = 0
for linede, lineen in zip(de, en):
    linede = linede.decode('unicode_escape').split()
    lineen = lineen.decode('unicode_escape').split()
    output.write('SENT:' + ' ' + str(index) + '\n')
    wordidx = 0
    for word in lineen:
        if tempatten[index][wordidx] == 0 or tempatten[index][wordidx]>len(linede):
            continue
        else:
            output.write('P'+' '+str(tempatten[index][wordidx]-1)+' '+str(wordidx)+'\n')
        wordidx += 1
    index += 1
