import os
s=os.listdir()
count=0
for i in s:
    try:
        count+=len(os.listdir(i))
    except:
        pass
print("Total number of Markets(District):",count)


