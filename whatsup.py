import os

# path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/whatsup/controlled_images"
path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/whatsup/controlled_clevr"
a = {}

for i in os.listdir(path):
    # print(i.split("_")[1])
    # exit(0)
    try:
        if i.split("_")[1] not in a:
            a[i.split("_")[1]] = 0
        else:
            a[i.split("_")[1]] +=1
    except:
        print(i)
        # exit(0)
print(a)