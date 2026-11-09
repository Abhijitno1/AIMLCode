import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")

'''
dblist = myclient.list_database_names()
print(dblist)
if "DirectoryMap" in dblist:
  print("The database exists.")
else:
  print("The database does not exist.")
'''

mydb = myclient["DirectoryMap"]
#print(mydb.list_collection_names())
mycol = mydb["DriveMaps"]
# Convert cursor to list to allow indexing and len()
#dbrecs = list(mycol.find({"IsFolder": False}, {"_id": 1, "FullPath": 1}).limit(10)) -- no more test sampling
dbrecs = list(mycol.find({"IsFolder": False}, {"_id": 1, "FullPath": 1}))
positive_pairs = []
negative_pairs = []
# Use enumerate to get the index of the current record
for current_index, rec in enumerate(dbrecs):
    # Find the last occurrence of backslash to separate directory from filename
    FullPath = rec["FullPath"]
    last_backslash_index = FullPath.rfind("\\")
    directory = FullPath[:last_backslash_index]
    filename = FullPath[last_backslash_index + 1:]
    positive_pairs.append({
        "file": filename,
        "folder": directory,
        "doesContain": 1
    })
    # Select any other record from dbrecs
    other_index = (current_index + 1) % len(dbrecs)
    other_rec = dbrecs[other_index]
    other_FullPath = other_rec["FullPath"]
    other_last_backslash_index = other_FullPath.rfind("\\")
    other_directory = other_FullPath[:other_last_backslash_index]
    other_filename = other_FullPath[other_last_backslash_index + 1:]
    negative_pairs.append({
        "file": filename,
        "folder": other_directory,
        "doesContain": 0
    })

# Export for use in training
__all__ = ["positive_pairs", "negative_pairs"]

'''
print("Positive Pairs:")
print(positive_pairs)
print("Negative Pairs:")
print(negative_pairs)
'''
