#!/usr/bin/env python


if __name__ == '__main__':
    with open("streams.txt") as f:
        temp=f.read().splitlines()
    print(temp)
    


for i in range(len(temp)):
    tempVal=temp[i]
    slashPosition=tempVal.find('/')
    tempVal3=tempVal[0:slashPosition]
    tempVal2=tempVal.replace(tempVal3,'')
    temp[i]=tempVal2
    print(temp[i])
    

with open("streams.txt",'w') as f:
    for i in range(len(temp)):
        f.write(temp[i]+"\n")
    
