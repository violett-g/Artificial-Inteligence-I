dict = {'key1':[('a','b',1),('1','2',1)]}
print(dict['key1'])

for x,y,w in dict['key1']:
    if y=='2':
        w+=1
        new = (x,y,w)
        dict['key1'].append(new)


print(dict)