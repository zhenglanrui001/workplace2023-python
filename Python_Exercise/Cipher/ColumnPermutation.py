
def Column_Permutation_Encription(text,key):
    dict1 = {}
    for i in range(len(key)):
        dict1[i+1] = []
    text = text.replace(" ","")
    for i in range(len(text)):
        index = (i + 1) % len(key)
        if index != 0:
            dict1[index].append(text[i])
        else:
            index = len(key)
            dict1[index].append(text[i])
    dict2 = {}
    index = 1
    for num in key:
        dict2[index] = dict1[num]
        index += 1
    print(dict2)

Column_Permutation_Encription("attack begins at five",[1,4,5,3,2,6])

def Column_Permutation_Decription(text,key):
    dict1 = {}
    for i in range(len(key)):
        dict1[i+1] = []
    text = text.replace(" ","")
    for i in range(len(text)):
        index = (i + 1) % len(key)
        if index != 0:
            dict1[index].append(text[i])
        else:
            index = len(key)
            dict1[index].append(text[i])
    dict2 = {}
    index = 1
    for num in key:
        dict2[index] = dict1[num]
        index += 1
    print(dict2)

Column_Permutation_Decription("abatgftetcnvaiikse",[1,4,5,3,2,6])