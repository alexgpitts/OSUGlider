

list1 = [1, 2, 3, 4]
list2 = [1, 2, 6, 4]
list3 = [1, 2, 3, 4]

if((any(i > 500 for i in list1) or any(i > 500 for i in list2) or any(i > 500 for i in list3))):
            print("bad data containing extremely large values")