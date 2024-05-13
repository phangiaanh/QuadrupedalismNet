import json


path = "/media/lap16043/fe6895ab-3146-48c7-ba28-d1d7add169ce/Thesis/Datasets/animal3d-20240404T061206Z-001/animal3d/test.json"

with open(path) as json_file:
  json_data = json.load(json_file)

# for i in json_data:
#   print(i)
#   print(type(json_data[i]))
#   if (type(json_data[i]) == list):
#     # print(len(json_data[i]))
#     print(json_data[i][0])
#     print(type(json_data[i][0]))

data = json_data['data'][0]
for i in data:
  print(i)
  # print(type(data[i]))
  if isinstance(data[i], list):
    print(len(data[i]))

print(len(json_data['data']))
  